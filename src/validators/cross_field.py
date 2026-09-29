"""Cross-field validators: completeness and arithmetic/temporal consistency."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from src.schemas.doc_types import REQUIRED_FIELDS
from src.schemas.document import DocumentType, ExtractedField, Finding
from src.validators.regex_rules import check_amounts, check_iso_dates, check_tax_id

MATH_TOLERANCE = 0.05


def by_name(fields: Iterable[ExtractedField]) -> dict[str, ExtractedField]:
    """Index fields by name (last one wins on duplicates)."""
    return {f.name: f for f in fields}


def check_required(fields: list[ExtractedField], doc_type: DocumentType) -> list[Finding]:
    """One ``missing_required`` error per required field without a value."""
    have = {f.name for f in fields if f.value not in (None, "")}
    return [
        Finding(
            severity="error",
            code="missing_required",
            field=name,
            message=f"required field '{name}' not found",
        )
        for name in REQUIRED_FIELDS.get(doc_type, [])
        if name not in have
    ]


def _num(field: ExtractedField | None) -> float | None:
    if field is None or not isinstance(field.value, int | float):
        return None
    return float(field.value)


def check_invoice_math(fields: list[ExtractedField]) -> list[Finding]:
    """``subtotal + tax == total`` within :data:`MATH_TOLERANCE`."""
    idx = by_name(fields)
    sub, tax, total = _num(idx.get("subtotal")), _num(idx.get("tax")), _num(idx.get("total"))
    if sub is None or tax is None or total is None:
        return []
    expected = round(sub + tax, 2)
    if abs(expected - total) > MATH_TOLERANCE:
        return [
            Finding(
                severity="error",
                code="math_mismatch",
                field="total",
                message="subtotal + tax does not equal total",
                details={"expected_total": expected, "got_total": total},
            )
        ]
    return []


def check_date_order(fields: list[ExtractedField]) -> list[Finding]:
    """``due_date`` must not precede ``issue_date``."""
    idx = by_name(fields)
    issue, due = idx.get("issue_date"), idx.get("due_date")
    if not (issue and due and isinstance(issue.value, str) and isinstance(due.value, str)):
        return []
    try:
        if date.fromisoformat(due.value) < date.fromisoformat(issue.value):
            return [
                Finding(
                    severity="error",
                    code="due_before_issue",
                    field="due_date",
                    message="due date precedes issue date",
                    details={"issue_date": issue.value, "due_date": due.value},
                )
            ]
    except ValueError:
        return []  # format problems are reported by check_iso_dates
    return []


def check_contract_term(fields: list[ExtractedField]) -> list[Finding]:
    """Contract term must be a positive number of months (<= 50 years)."""
    term = by_name(fields).get("term_months")
    if term is None or not isinstance(term.value, int | float):
        return []
    if term.value <= 0 or term.value > 600:
        return [
            Finding(
                severity="warn",
                code="implausible_term",
                field="term_months",
                message="contract term outside 1-600 months",
                details={"value": float(term.value)},
            )
        ]
    return []


def validate_fields(fields: list[ExtractedField], doc_type: DocumentType) -> list[Finding]:
    """Run every validator relevant to ``doc_type`` and return all findings."""
    findings: list[Finding] = []
    findings += check_required(fields, doc_type)
    findings += check_iso_dates(fields)
    findings += check_amounts(fields)
    findings += check_tax_id(fields)
    if doc_type == "invoice":
        findings += check_invoice_math(fields)
        findings += check_date_order(fields)
    if doc_type == "contract":
        findings += check_contract_term(fields)
    return findings
