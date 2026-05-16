"""Cross-field validators — math consistency, completeness, format checks."""
from __future__ import annotations

import re
from typing import Any, Iterable

from src.api.schemas import ExtractedField
from src.extractors.schemas_by_type import REQUIRED_FIELDS


def _by_name(fields: Iterable[ExtractedField]) -> dict[str, ExtractedField]:
    return {f.name: f for f in fields}


def check_required(fields: list[ExtractedField], doc_type: str) -> list[dict[str, Any]]:
    """Returns list of findings for missing required fields."""
    required = set(REQUIRED_FIELDS.get(doc_type, []))
    have = {f.name for f in fields if f.value not in (None, "")}
    missing = required - have
    return [{"severity": "error", "code": "missing_required", "field": name}
            for name in sorted(missing)]


def check_iso_dates(fields: list[ExtractedField]) -> list[dict[str, Any]]:
    iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    findings = []
    for f in fields:
        if f.name.endswith("_date") and isinstance(f.value, str) and not iso.match(f.value):
            findings.append({"severity": "warn", "code": "non_iso_date",
                              "field": f.name, "value": f.value})
    return findings


def check_invoice_math(fields: list[ExtractedField]) -> list[dict[str, Any]]:
    by = _by_name(fields)
    findings = []
    sub = by.get("subtotal")
    tax = by.get("tax")
    total = by.get("total")
    try:
        if sub and tax and total and all(isinstance(x.value, (int, float)) for x in (sub, tax, total)):
            expected = round(float(sub.value) + float(tax.value), 2)
            if abs(expected - float(total.value)) > 0.05:
                findings.append({
                    "severity": "error", "code": "math_mismatch",
                    "expected_total": expected, "got_total": float(total.value),
                })
    except (TypeError, ValueError):
        pass
    return findings


def overall_confidence(fields: list[ExtractedField]) -> float:
    if not fields:
        return 0.0
    return sum(f.confidence for f in fields) / len(fields)


def validate_all(fields: list[ExtractedField], doc_type: str) -> tuple[float, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    findings += check_required(fields, doc_type)
    findings += check_iso_dates(fields)
    if doc_type == "invoice":
        findings += check_invoice_math(fields)
    # Penalize confidence by 0.1 per error finding
    base = overall_confidence(fields)
    n_errors = sum(1 for f in findings if f["severity"] == "error")
    return max(0.0, base - 0.1 * n_errors), findings


def routing_decision(confidence: float, *, auto_thr: float, review_thr: float) -> str:
    if confidence >= auto_thr:
        return "auto_approve"
    if confidence >= review_thr:
        return "human_review"
    return "vlm_fallback"
