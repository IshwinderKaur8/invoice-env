import random
from typing import Tuple, Dict, Any, Optional
from .models import InvoiceObservation, InvoiceAction, InvoiceReward
from .dataset import load_invoices
from .graders import grade_extraction, grade_category, grade_anomaly


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

    def __init__(self, batch_size: int = 10, seed: Optional[int] = None, shuffle: bool = True, logger: Optional[Any] = None):
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
        if seed is not None:
            random.seed(seed)

        self.invoices: list[Dict[str, Any]] = []
        self.pointer: int = 0
        self.current_invoice: Optional[Dict[str, Any]] = None
        self.total_reward: float = 0.0
        self.steps: int = 0

    def reset(self) -> InvoiceObservation:
        """
        Reset environment to start a new episode.
        Returns:
            InvoiceObservation: initial observation containing vendor_name,
            invoice_date, amount, description, and metadata.
        """
        dataset = load_invoices()
        self.invoices = random.sample(dataset, self.batch_size) if self.shuffle else dataset[:self.batch_size]
        self.pointer = 0
        self.total_reward = 0.0
        self.steps = 0
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
        invoice = self.current_invoice

        # Deterministic grading
        extraction_score = grade_extraction(action.extracted_fields, invoice)
        category_score = grade_category(action.category, invoice)
        anomaly_score = grade_anomaly(action.anomaly_flag, invoice)

        # Weighted reward
        total_score = (
            0.4 * extraction_score +
            0.3 * category_score +
            0.3 * anomaly_score
        )

        reward = InvoiceReward(
            score=total_score,
            details={
                "extraction": extraction_score,
                "category": category_score,
                "anomaly": anomaly_score
            }
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
                vendor_name="",
                invoice_date="",
                amount=0.0,
                description="",
                metadata={"terminal": True}
            )

        # Rich info dictionary
        info = {
            "invoice_id": invoice.get("id"),
            "ground_truth_category": invoice.get("category"),
            "ground_truth_anomaly": invoice.get("anomaly_flag"),
            "step": self.steps,
            "cumulative_reward": self.total_reward
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
            "steps": self.steps,
            "current_invoice": self.current_invoice
        }

    def _make_observation(self, invoice: Dict[str, Any]) -> InvoiceObservation:
        """Helper to convert raw invoice dict into typed Observation."""
        return InvoiceObservation(
            vendor_name=invoice["vendor_name"],
            invoice_date=invoice["invoice_date"],
            amount=invoice["amount"],
            description=invoice["description"],
            metadata={"id": self.pointer}
        )