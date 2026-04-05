from typing import Optional, Dict, Any
from pydantic import BaseModel, validator
import re
from datetime import datetime


class InvoiceObservation(BaseModel):
    """
    Observation model for invoice environment.
    Fields:
    vendor_name (str): name of vendor (e.g., Amazon, Uber)
    invoice_date (str): date in YYYY-MM-DD format
    amount (float): invoice amount (non-negative)
    description (str): textual description of expense
    metadata (dict): auxiliary info (e.g., invoice id)
    """
    vendor_name: str
    invoice_date: str
    amount: float
    description: str
    metadata: Dict[str, Any]

    @validator("vendor_name")
    def vendor_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Vendor name cannot be empty")
        return v

    @validator("invoice_date")
    def valid_date_format(cls, v):
        # Accepts YYYY-MM-DD format
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invoice date must be in YYYY-MM-DD format")
        return v

    @validator("amount")
    def non_negative_amount(cls, v):
        if v < 0:
            raise ValueError("Invoice amount must be non-negative")
        return v


class InvoiceAction(BaseModel):
    """
    Action model for agent response.
    Fields:
    extracted_fields (dict): must include vendor_name and invoice_date
    category (Optional[str]): one of {Travel, Office Supplies, Utilities, Misc}
    anomaly_flag (Optional[bool]): True if anomaly detected
    """
    extracted_fields: Dict[str, str]
    category: Optional[str] = None
    anomaly_flag: Optional[bool] = None

    @validator("extracted_fields")
    def must_contain_required_fields(cls, v):
        required = {"vendor_name", "invoice_date"}
        missing = required - set(v.keys())
        if missing:
            raise ValueError(f"Missing required extracted fields: {missing}")
        return v

    @validator("category")
    def valid_category(cls, v):
        if v is None:
            return v
        allowed = {"Travel", "Office Supplies", "Utilities", "Misc"}
        if v not in allowed:
            raise ValueError(f"Category must be one of {allowed}")
        return v


class InvoiceReward(BaseModel):
    """
    Reward model for environment feedback.
    Fields:
    score (float): continuous reward between 0.0 and 1.0
    details (dict): breakdown of extraction, category, anomaly scores
    """
    score: float
    details: Dict[str, Any]

    @validator("score")
    def score_in_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Reward score must be between 0.0 and 1.0")
        return v