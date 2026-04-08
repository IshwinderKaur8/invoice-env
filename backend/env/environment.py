import random
from typing import Any, Dict, List, Optional, Tuple

from .dataset import load_invoices
from .graders import detection_metrics, grade_anomaly, grade_category, grade_extraction
from .models import InvoiceAction, InvoiceObservation, InvoiceReward
from .tasks import TASKS, compute_weighted_reward


MIN_REWARD_SCORE = 0.01
MAX_REWARD_SCORE = 0.99


class InvoiceEnv:
    """
    Invoice & Receipt Processing Environment (OpenEnv-compliant).

    Agents interact with this environment to learn:
      - Task 1: Field Extraction (vendor_name, invoice_date)
      - Task 2: Expense Categorization (Travel, Office Supplies, Utilities, Misc)
      - Task 3: Anomaly Detection (duplicate invoices, unusually high amounts)

    OpenEnv Specification:
      - step(action) -> (observation, reward, done, info)
      - reset() -> initial observation
      - state() -> current environment state

    Episode:
      One episode = batch of invoices (default batch_size=10).
    """

    def __init__(
        self,
        batch_size: int = 10,
        seed: Optional[int] = None,
        shuffle: bool = True,
        logger: Optional[Any] = None,
    ):
        """
        Args:
            batch_size: number of invoices per episode
            seed: random seed for reproducibility
            shuffle: whether to shuffle invoices
            logger: optional logging hook (e.g., TensorBoard, print)
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.logger = logger
        self.seed = seed
        self._rng = random.Random(seed)

        self.invoices: List[Dict[str, Any]] = []
        self.pointer: int = 0
        self.current_invoice: Optional[Dict[str, Any]] = None
        self.total_reward: float = 0.0
        self.steps: int = 0
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.last_action_signature: Optional[Tuple[Any, ...]] = None
        self.repeat_action_streak: int = 0
        self.loop_events: int = 0
        self.destructive_events: int = 0

    def reset(self) -> InvoiceObservation:
        """
        Reset environment to start a new episode.
        Returns:
            InvoiceObservation: initial observation containing vendor_name,
            invoice_date, amount, description, and metadata.
        """
        dataset = load_invoices()
        if self.batch_size > len(dataset):
            raise ValueError(f"batch_size {self.batch_size} exceeds dataset size {len(dataset)}")

        self.invoices = self._rng.sample(dataset, self.batch_size) if self.shuffle else dataset[: self.batch_size]
        self.pointer = 0
        self.total_reward = 0.0
        self.steps = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.last_action_signature = None
        self.repeat_action_streak = 0
        self.loop_events = 0
        self.destructive_events = 0
        self.current_invoice = self.invoices[self.pointer]
        return self._make_observation(self.current_invoice)

    def step(self, action: InvoiceAction) -> Tuple[InvoiceObservation, InvoiceReward, bool, Dict]:
        """
        Apply agent's action to current invoice.
        Args:
            action (InvoiceAction): agent's structured response including
            extracted_fields, category, anomaly_flag.
        Returns:
            Tuple:
              - InvoiceObservation: next invoice (or terminal observation if done)
              - InvoiceReward: continuous reward with task-level details
              - done (bool): True if episode finished
              - info (dict): ground truth labels and episode metadata
        """
        if self.current_invoice is None:
            raise RuntimeError("Environment is not initialized. Call reset() before step().")
        if not isinstance(action, InvoiceAction):
            action = InvoiceAction(**action)

        invoice = self.current_invoice

        # Deterministic grading
        extraction_score = grade_extraction(action.extracted_fields, invoice)
        category_score = grade_category(action.category, invoice)
        try:
            anomaly_score = grade_anomaly(
                action.anomaly_flag,
                invoice,
                tp=self.tp,
                fp=self.fp,
                fn=self.fn,
            )
        except TypeError:
            # Backward compatibility for monkeypatched 2-argument graders in tests.
            anomaly_score = grade_anomaly(action.anomaly_flag, invoice)

        truth_anomaly = bool(invoice.get("anomaly_flag", False))
        pred_anomaly = bool(action.anomaly_flag) if action.anomaly_flag is not None else False

        self.tp += int(pred_anomaly and truth_anomaly)
        self.fp += int(pred_anomaly and not truth_anomaly)
        self.fn += int((not pred_anomaly) and truth_anomaly)

        missing_fields = sum(
            1
            for key in ("vendor_name", "invoice_date")
            if not str(action.extracted_fields.get(key, "") or "").strip()
        )

        action_signature = self._signature(action)
        if action_signature == self.last_action_signature:
            self.repeat_action_streak += 1
        else:
            self.repeat_action_streak = 1
            self.last_action_signature = action_signature

        loop_penalty = 0.12 if self.repeat_action_streak >= 3 else 0.0
        if loop_penalty > 0:
            self.loop_events += 1

        destructive_action = self._is_destructive_action(action, missing_fields)
        destructive_penalty = 0.18 if destructive_action else 0.0
        if destructive_action:
            self.destructive_events += 1

        reward_parts = compute_weighted_reward(
            extraction_score=extraction_score,
            category_score=category_score,
            anomaly_score=anomaly_score,
            missing_fields=missing_fields,
            false_anomaly=pred_anomaly and not truth_anomaly,
            missed_anomaly=(not pred_anomaly) and truth_anomaly,
        )

        total_score = max(
            MIN_REWARD_SCORE,
            min(MAX_REWARD_SCORE, reward_parts["final_score"] - loop_penalty - destructive_penalty),
        )

        reward = InvoiceReward(
            score=total_score,
            details={
                "extraction": extraction_score,
                "category": category_score,
                "anomaly": anomaly_score,
                "field_extraction": extraction_score,
                "expense_categorization": category_score,
                "anomaly_detection": anomaly_score,
                "task_scores": {
                    "field_extraction": extraction_score,
                    "expense_categorization": category_score,
                    "anomaly_detection": anomaly_score,
                },
                "base_score": reward_parts["base_score"],
                "penalty": reward_parts["penalty"],
                "loop_penalty": loop_penalty,
                "destructive_penalty": destructive_penalty,
                "missing_fields": missing_fields,
            },
        )

        # Update metrics
        self.total_reward += total_score
        self.steps += 1

        # Advance pointer
        self.pointer += 1
        done = self.pointer >= len(self.invoices)

        if not done:
            self.current_invoice = self.invoices[self.pointer]
            next_obs = self._make_observation(self.current_invoice)
        else:
            # Return a terminal observation for RL loop cleanliness
            next_obs = InvoiceObservation(
                vendor_name="__terminal__",
                invoice_date="1970-01-01",
                amount=0.0,
                description="Terminal observation",
                metadata={"terminal": True},
            )

        metrics = detection_metrics(self.tp, self.fp, self.fn)

        # Rich info dictionary
        info = {
            "invoice_id": invoice.get("id"),
            "ground_truth_category": invoice.get("category"),
            "ground_truth_anomaly": invoice.get("anomaly_flag"),
            "task_scores": {
                "field_extraction": extraction_score,
                "expense_categorization": category_score,
                "anomaly_detection": anomaly_score,
            },
            "task_graders": {
                "field_extraction": "env.graders.grade_extraction",
                "expense_categorization": "env.graders.grade_category",
                "anomaly_detection": "env.graders.grade_anomaly",
            },
            "task_context": {
                "task_1_observation": "Raw invoice text for field extraction",
                "task_2_observation": "Invoice metadata for categorization",
                "task_3_observation": "Batch-level anomaly statistics",
            },
            "step": self.steps,
            "cumulative_reward": self.total_reward,
            "anomaly_precision": metrics["precision"],
            "anomaly_recall": metrics["recall"],
            "anomaly_f1": metrics["f1"],
        }

        # Optional logging
        if self.logger:
            self.logger.log({
                "step": self.steps,
                "reward": reward.score,
                "details": reward.details
            })

        return next_obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """
        Get current environment state for debugging/logging.
        Returns:
            dict: includes pointer, remaining invoices, cumulative reward,
            steps taken, and current invoice.
        """
        return {
            "pointer": self.pointer,
            "remaining": len(self.invoices) - self.pointer,
            "total_reward": self.total_reward,
            "average_reward": (self.total_reward / self.steps) if self.steps else 0.0,
            "steps": self.steps,
            "current_invoice": self.current_invoice,
            "anomaly_counts": {"tp": self.tp, "fp": self.fp, "fn": self.fn},
            "trajectory_events": {
                "repeat_action_streak": self.repeat_action_streak,
                "loop_events": self.loop_events,
                "destructive_events": self.destructive_events,
            },
            "tasks": [
                {
                    "id": task.id,
                    "name": task.name,
                    "difficulty": task.difficulty,
                    "description": task.description,
                    "grader": f"{task.grader.__module__}.{task.grader.__name__}",
                    "graders": [f"{fn.__module__}.{fn.__name__}" for fn in task.graders],
                }
                for task in TASKS
            ],
        }

    def _make_observation(self, invoice: Dict[str, Any]) -> InvoiceObservation:
        """Helper to convert raw invoice dict into typed Observation."""
        raw_text = (
            f"Vendor: {invoice['vendor_name']} | "
            f"Date: {invoice['invoice_date']} | "
            f"Amount: {invoice['amount']} | "
            f"Description: {invoice['description']}"
        )
        return InvoiceObservation(
            vendor_name=invoice["vendor_name"],
            invoice_date=invoice["invoice_date"],
            amount=invoice["amount"],
            description=invoice["description"],
            metadata={
                "id": invoice["id"],
                "invoice_ref": invoice["invoice_ref"],
                "raw_text": raw_text,
                "line_items": invoice.get("line_items", []),
                "anomaly_type": invoice.get("anomaly_type", "none"),
            },
        )

    @staticmethod
    def _signature(action: InvoiceAction) -> Tuple[Any, ...]:
        return (
            tuple(sorted(action.extracted_fields.items())),
            action.category,
            action.anomaly_flag,
        )

    @staticmethod
    def _is_destructive_action(action: InvoiceAction, missing_fields: int) -> bool:
        if missing_fields == 2 and not action.category and action.anomaly_flag is None:
            return True

        raw_values = [
            str(action.extracted_fields.get("vendor_name", "") or "").lower(),
            str(action.extracted_fields.get("invoice_date", "") or "").lower(),
        ]
        unsafe_markers = ("drop table", "delete all", "rm -rf", "__terminal__")
        return any(marker in value for value in raw_values for marker in unsafe_markers)