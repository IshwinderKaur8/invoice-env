from __future__ import annotations

from typing import Any, Dict, Optional

from rapidfuzz import fuzz


def _normalize(value: str) -> str:
	return value.strip().lower()


def _clamp_open_unit(value: float) -> float:
    # Keep all task scores strictly within (0,1) to satisfy validator constraints.
    return max(0.01, min(0.99, float(value)))


def grade_extraction(extracted_fields: Dict[str, Any], invoice: Dict[str, Any]) -> float:
    # Support validators that pass the full action payload as first arg.
    if isinstance(extracted_fields, dict) and "extracted_fields" in extracted_fields:
        nested = extracted_fields.get("extracted_fields")
        if isinstance(nested, dict):
            extracted_fields = nested

    if not isinstance(extracted_fields, dict):
        extracted_fields = {}
    if not isinstance(invoice, dict):
        invoice = {}

    required = ("vendor_name", "invoice_date")
    per_field_scores = []

    for key in required:
        pred = str(extracted_fields.get(key, "") or "")
        truth = str(invoice.get(key, "") or "")

        if _normalize(pred) == _normalize(truth):
            per_field_scores.append(0.99)  # almost perfect
            continue

        ratio = fuzz.ratio(_normalize(pred), _normalize(truth))
        if ratio >= 80:
            per_field_scores.append(round(min(0.98, ratio / 100.0), 4))  # partial but <1
        else:
            per_field_scores.append(0.01)  # minimal, not zero

    return _clamp_open_unit(round(sum(per_field_scores) / len(required), 4))


def grade_category(predicted_category: Optional[str], invoice: Dict[str, Any]) -> float:
    # Support validators that pass the full action payload as first arg.
    if isinstance(predicted_category, dict):
        predicted_category = predicted_category.get("category")

    if not isinstance(invoice, dict):
        invoice = {}

    truth = invoice.get("category")
    if predicted_category is None:
        return 0.01

    tokens = [piece.strip() for piece in predicted_category.replace("|", ",").split(",") if piece.strip()]
    if not tokens:
        return 0.01

    if tokens[0] == truth:
        return _clamp_open_unit(0.99)
    if truth in tokens[:2]:
        return _clamp_open_unit(0.5)
    close_pairs = {
        frozenset(("Travel", "Misc")),
        frozenset(("Office Supplies", "Misc")),
        frozenset(("Utilities", "Misc")),
    }
    if frozenset((tokens[0], truth)) in close_pairs:
        return _clamp_open_unit(0.5)
    return _clamp_open_unit(0.01)



def detection_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
	"""Compute precision, recall, and F1 for anomaly detection."""
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	if precision + recall == 0:
		f1 = 0.0
	else:
		f1 = 2 * precision * recall / (precision + recall)
	return {
		"precision": round(precision, 4),
		"recall": round(recall, 4),
		"f1": round(f1, 4),
	}


def grade_anomaly(
    predicted_flag: Optional[bool],
    invoice: Dict[str, Any],
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
    **_: Any,
) -> float:
    # Support validators that pass the full action payload as first arg.
    if isinstance(predicted_flag, dict):
        predicted_flag = predicted_flag.get("anomaly_flag")

    if not isinstance(invoice, dict):
        invoice = {}

    truth = bool(invoice.get("anomaly_flag", False))
    pred = bool(predicted_flag) if predicted_flag is not None else False

    next_tp = tp + int(pred and truth)
    next_fp = fp + int(pred and not truth)
    next_fn = fn + int((not pred) and truth)

    f1 = detection_metrics(next_tp, next_fp, next_fn)["f1"]
    # Clamp into (0,1)
    if f1 <= 0.0:
        return _clamp_open_unit(0.01)
    if f1 >= 1.0:
        return _clamp_open_unit(0.99)
    return _clamp_open_unit(f1)

