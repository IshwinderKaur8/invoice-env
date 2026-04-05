import pytest
from pydantic import ValidationError
from env.models import InvoiceObservation, InvoiceAction, InvoiceReward


def test_valid_observation():
    obs = InvoiceObservation(
        vendor_name="Uber",
        invoice_date="2026-03-01",
        amount=45.0,
        description="Taxi ride",
        metadata={"id": 0}
    )
    assert obs.vendor_name == "Uber"
    assert obs.amount == 45.0


def test_invalid_date_format():
    with pytest.raises(ValidationError):
        InvoiceObservation(
            vendor_name="Uber",
            invoice_date="03-01-2026",  # wrong format
            amount=45.0,
            description="Taxi ride",
            metadata={"id": 0}
        )


def test_negative_amount():
    with pytest.raises(ValidationError):
        InvoiceObservation(
            vendor_name="Uber",
            invoice_date="2026-03-01",
            amount=-10.0,
            description="Taxi ride",
            metadata={"id": 0}
        )


def test_action_missing_fields():
    with pytest.raises(ValidationError):
        InvoiceAction(
            extracted_fields={"vendor_name": "Uber"},  # missing invoice_date
            category="Travel",
            anomaly_flag=False
        )


def test_action_invalid_category():
    with pytest.raises(ValidationError):
        InvoiceAction(
            extracted_fields={"vendor_name": "Uber", "invoice_date": "2026-03-01"},
            category="Food",  # not allowed
            anomaly_flag=False
        )


def test_valid_reward():
    reward = InvoiceReward(score=0.85, details={"extraction": 1.0})
    assert reward.score == 0.85


def test_invalid_reward_score():
    with pytest.raises(ValidationError):
        InvoiceReward(score=1.5, details={"extraction": 1.0})