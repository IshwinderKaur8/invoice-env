import json
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openai import OpenAI

from backend.db.mongo import get_invoices_collection, get_runs_collection
from env.dataset import load_invoices
from env.environment import InvoiceEnv
from env.models import InvoiceAction


SYSTEM_PROMPT = (
    "You are an invoice processing assistant. "
    "Return ONLY valid JSON with keys: extracted_fields, category, anomaly_flag. "
    "extracted_fields must contain vendor_name and invoice_date. "
    "category must be one of Travel, Office Supplies, Utilities, Misc."
)


class EnvSession:
    def __init__(self) -> None:
        self.env: Optional[InvoiceEnv] = None
        self.latest_observation: Optional[Dict[str, Any]] = None
        self.last_step_result: Optional[Dict[str, Any]] = None
        self.loaded_batch: List[Dict[str, Any]] = []


SESSION = EnvSession()
IN_MEMORY_RUNS: List[Dict[str, Any]] = []


def _serialize_invoice(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(doc.get("id")),
        "invoice_ref": str(doc.get("invoice_ref", "")),
        "vendor_name": str(doc.get("vendor_name", "")),
        "invoice_date": str(doc.get("invoice_date", "")),
        "amount": float(doc.get("amount", 0.0)),
        "description": str(doc.get("description", "")),
        "line_items": doc.get("line_items", []),
        "category": str(doc.get("category", "Misc")),
        "anomaly_flag": bool(doc.get("anomaly_flag", False)),
        "anomaly_type": str(doc.get("anomaly_type", "none")),
    }


def _ensure_invoice_seed_data() -> None:
    invoices_col = get_invoices_collection()
    if invoices_col.count_documents({}) > 0:
        return

    records = load_invoices()
    invoices_col.insert_many(records)


def _load_batch_from_mongo(batch_size: int) -> List[Dict[str, Any]]:
    try:
        _ensure_invoice_seed_data()
        invoices_col = get_invoices_collection()
        sampled = list(invoices_col.aggregate([{"$sample": {"size": batch_size}}]))
        return [_serialize_invoice(doc) for doc in sampled]
    except Exception:
        local_dataset = load_invoices()
        sampled_local = random.sample(local_dataset, min(batch_size, len(local_dataset)))
        return [_serialize_invoice(doc) for doc in sampled_local]


def reset_environment(batch_size: int = 12) -> Dict[str, Any]:
    batch = _load_batch_from_mongo(batch_size=batch_size)

    env = InvoiceEnv(batch_size=len(batch), shuffle=False, seed=42)
    env.invoices = batch
    env.pointer = 0
    env.total_reward = 0.0
    env.steps = 0
    env.tp = 0
    env.fp = 0
    env.fn = 0
    env.current_invoice = batch[0]

    observation = env._make_observation(batch[0])

    SESSION.env = env
    SESSION.latest_observation = observation.model_dump()
    SESSION.last_step_result = None
    SESSION.loaded_batch = batch

    return {
        "observation": SESSION.latest_observation,
        "state": env.state(),
    }


def step_environment(action_payload: Dict[str, Any]) -> Dict[str, Any]:
    if SESSION.env is None:
        reset_environment()

    assert SESSION.env is not None
    action = InvoiceAction(**action_payload)
    next_obs, reward, done, info = SESSION.env.step(action)

    result = {
        "observation": next_obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }
    SESSION.latest_observation = result["observation"]
    SESSION.last_step_result = result
    return result


def get_state() -> Dict[str, Any]:
    if SESSION.env is None:
        reset_environment()

    assert SESSION.env is not None
    return {
        "state": SESSION.env.state(),
        "latest_observation": SESSION.latest_observation,
        "last_step_result": SESSION.last_step_result,
    }


def _extract_json(text: str) -> Dict[str, Any]:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.strip("`")
        payload = payload.replace("json", "", 1).strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(payload[start : end + 1])
        raise


def _heuristic_action(observation: Dict[str, Any], seen_refs: set[str]) -> Dict[str, Any]:
    metadata = observation.get("metadata", {})
    invoice_ref = str(metadata.get("invoice_ref", "")).strip()
    amount = float(observation.get("amount", 0.0))
    vendor = str(observation.get("vendor_name", "")).lower()

    duplicate_flag = bool(invoice_ref and invoice_ref in seen_refs)
    if invoice_ref:
        seen_refs.add(invoice_ref)

    if any(token in vendor for token in ("uber", "lyft", "airlines", "marriott")):
        category = "Travel"
    elif any(token in vendor for token in ("amazon", "staples", "ikea")):
        category = "Office Supplies"
    elif any(token in vendor for token in ("electricity", "water", "internet", "gas")):
        category = "Utilities"
    else:
        category = "Misc"

    return {
        "extracted_fields": {
            "vendor_name": str(observation.get("vendor_name", "")),
            "invoice_date": str(observation.get("invoice_date", "")),
        },
        "category": category,
        "anomaly_flag": duplicate_flag or amount > 2500,
    }


def _openai_action(client: OpenAI, model_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (
        "Process this invoice and return JSON only.\\n"
        f"Vendor: {observation['vendor_name']}\\n"
        f"Invoice Date: {observation['invoice_date']}\\n"
        f"Amount: {observation['amount']}\\n"
        f"Description: {observation['description']}\\n"
        f"Metadata: {json.dumps(observation.get('metadata', {}))}"
    )
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content or "{}"
    return _extract_json(content)


def run_agent_full(batch_size: int = 12, mode: str = "auto") -> Dict[str, Any]:
    reset_environment(batch_size=batch_size)
    assert SESSION.env is not None

    mode = mode.lower().strip()
    if mode not in {"auto", "openai", "heuristic"}:
        raise ValueError("mode must be one of auto, openai, heuristic")

    api_key = os.getenv("OPENAI_API_KEY")
    use_openai = mode == "openai" or (mode == "auto" and bool(api_key))
    if mode == "openai" and not api_key:
        raise ValueError("OPENAI_API_KEY required for openai mode")

    client = OpenAI(api_key=api_key) if use_openai and api_key else None
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    done = False
    step_results: List[Dict[str, Any]] = []
    seen_refs: set[str] = set()

    while not done:
        assert SESSION.latest_observation is not None
        obs = SESSION.latest_observation

        if client is not None:
            try:
                action = _openai_action(client, model_name, obs)
            except Exception:
                action = _heuristic_action(obs, seen_refs)
        else:
            action = _heuristic_action(obs, seen_refs)

        if isinstance(action.get("category"), list):
            action["category"] = "|".join(str(item) for item in action["category"][:2])

        step_result = step_environment(action)
        step_results.append(
            {
                "observation": obs,
                "action": action,
                "reward": step_result["reward"],
                "done": step_result["done"],
                "info": step_result["info"],
            }
        )
        done = step_result["done"]

    final_score = SESSION.env.total_reward / SESSION.env.steps if SESSION.env.steps else 0.0
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    run_doc = {
        "run_id": run_id,
        "mode": "openai" if client is not None else "heuristic",
        "results": step_results,
        "final_score": final_score,
        "steps": SESSION.env.steps,
        "timestamp": timestamp,
    }
    try:
        get_runs_collection().insert_one(run_doc)
    except Exception:
        # PyMongo may inject _id even when insert fails; strip before storing fallback data.
        run_doc.pop("_id", None)
        IN_MEMORY_RUNS.insert(0, run_doc)

    # Always return JSON-safe payload.
    run_doc.pop("_id", None)

    return run_doc


def get_results(limit: int = 20) -> Dict[str, Any]:
    try:
        cursor = get_runs_collection().find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        runs = list(cursor)
    except Exception:
        runs = IN_MEMORY_RUNS[:limit]

    return {
        "latest_run": runs[0] if runs else None,
        "runs": runs,
    }
