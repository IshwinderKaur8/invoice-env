from __future__ import annotations

from typing import Any, Dict, Optional

from rapidfuzz import fuzz


def _normalize(value: str) -> str:
	return value.strip().lower()


def _clamp_open_unit(value: float) -> float:
	# Keep task scores strictly inside (0, 1) for validator compatibility.
	return max(0.001, min(0.999, float(value)))


def _coerce_invoice(invoice: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
	if isinstance(invoice, dict):
		return invoice
	for key in ("invoice", "observation", "ground_truth", "target", "label"):
		value = kwargs.get(key)
		if isinstance(value, dict):
			return value
	return {}


def grade_extraction(extracted_fields: Any = None, invoice: Any = None, *args: Any, **kwargs: Any) -> float:
	"""
	Grade field extraction for vendor_name and invoice_date.
	Exact match: 0.999, fuzzy (>80): partial score, otherwise 0.001.
	"""
	# Support validators that pass action payload under different names.
	if extracted_fields is None:
		extracted_fields = kwargs.get("action") or kwargs.get("prediction") or kwargs.get("output")
	if isinstance(extracted_fields, dict) and "extracted_fields" in extracted_fields:
		nested = extracted_fields.get("extracted_fields")
		if isinstance(nested, dict):
			extracted_fields = nested

	if invoice is None and args:
		invoice = args[0]

	if not isinstance(extracted_fields, dict):
		extracted_fields = {}
	invoice = _coerce_invoice(invoice, kwargs)

	required = ("vendor_name", "invoice_date")
	per_field_scores = []

	for key in required:
		pred = str(extracted_fields.get(key, "") or "")
		truth = str(invoice.get(key, "") or "")

		if _normalize(pred) == _normalize(truth):
			per_field_scores.append(0.999)
		else:
			ratio = fuzz.ratio(_normalize(pred), _normalize(truth))
			if ratio >= 80:
				per_field_scores.append(min(0.998, ratio / 100.0))
			else:
				per_field_scores.append(0.001)

	score = sum(per_field_scores) / len(required)
	return _clamp_open_unit(score)


def grade_category(predicted_category: Any = None, invoice: Any = None, *args: Any, **kwargs: Any) -> float:
	"""
	Grade category classification.
	Correct: 1.0, close guess: 0.5, otherwise 0.0.
	Also accepts optional top-2 format "A|B" where second position gets 0.5.
	"""
	# Support validators that pass action payload under different names.
	if predicted_category is None:
		predicted_category = kwargs.get("action") or kwargs.get("prediction") or kwargs.get("output")
	if isinstance(predicted_category, dict):
		predicted_category = predicted_category.get("category")

	if invoice is None and args:
		invoice = args[0]

	invoice = _coerce_invoice(invoice, kwargs)

	truth = invoice.get("category")
	if predicted_category is None:
		return _clamp_open_unit(0.001)
	if not isinstance(predicted_category, str):
		predicted_category = str(predicted_category)

	tokens = [piece.strip() for piece in predicted_category.replace("|", ",").split(",") if piece.strip()]
	if not tokens:
		return _clamp_open_unit(0.001)

	if tokens[0] == truth:
		return _clamp_open_unit(0.999)
	if truth in tokens[:2]:
		return _clamp_open_unit(0.5)

	close_pairs = {
		frozenset(("Travel", "Misc")),
		frozenset(("Office Supplies", "Misc")),
		frozenset(("Utilities", "Misc")),
	}
	if frozenset((tokens[0], truth)) in close_pairs:
		return _clamp_open_unit(0.5)
	return _clamp_open_unit(0.001)


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
	predicted_flag: Any = None,
	invoice: Any = None,
	tp: int = 0,
	fp: int = 0,
	fn: int = 0,
	*args: Any,
	**kwargs: Any,
) -> float:
	"""
	Grade anomaly detection using F1 over running confusion counts.
	"""
	# Support validators that pass action payload under different names.
	if predicted_flag is None:
		predicted_flag = kwargs.get("action") or kwargs.get("prediction") or kwargs.get("output")
	if isinstance(predicted_flag, dict):
		predicted_flag = predicted_flag.get("anomaly_flag")

	if invoice is None and args:
		invoice = args[0]
	invoice = _coerce_invoice(invoice, kwargs)

	try:
		tp = int(tp or 0)
		fp = int(fp or 0)
		fn = int(fn or 0)
	except (TypeError, ValueError):
		tp, fp, fn = 0, 0, 0

	truth = bool(invoice.get("anomaly_flag", False))
	pred = bool(predicted_flag) if predicted_flag is not None else False

	next_tp = tp + int(pred and truth)
	next_fp = fp + int(pred and not truth)
	next_fn = fn + int((not pred) and truth)

	f1 = detection_metrics(next_tp, next_fp, next_fn)["f1"]
	if f1 <= 0.0:
		return _clamp_open_unit(0.001)
	if f1 >= 1.0:
		return _clamp_open_unit(0.999)
	return _clamp_open_unit(f1)
