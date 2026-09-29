"""PII redaction for extracted values and log lines.

Medical and HR forms carry identifiers that must not leak into logs, the
database or the demo. Redaction is regex-based (no model) and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas.document import ExtractedField

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "phone": re.compile(
        r"(?<![\d-])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"
    ),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b"),
}

# Fields whose values are legitimately numeric identifiers and must not be masked.
SAFE_FIELDS = frozenset(
    {
        "invoice_number",
        "receipt_number",
        "contract_number",
        "form_id",
        "vendor_tax_id",
        "total",
        "subtotal",
        "tax",
        "total_value",
        "term_months",
        "issue_date",
        "due_date",
        "effective_date",
        "date",
        "fiscal_year",
        "currency",
    }
)


@dataclass
class RedactionReport:
    """What was redacted and how many times per category."""

    counts: dict[str, int]

    @property
    def total(self) -> int:
        """Total number of masked spans."""
        return sum(self.counts.values())


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_text(text: str) -> tuple[str, RedactionReport]:
    """Mask PII spans in free text with ``[REDACTED:<kind>]`` tokens.

    Credit-card candidates are only masked when they pass the Luhn check, so
    invoice numbers and totals survive.
    """
    counts: dict[str, int] = {}

    def _mask(kind: str, m: re.Match[str]) -> str:
        counts[kind] = counts.get(kind, 0) + 1
        return f"[REDACTED:{kind}]"

    out = PII_PATTERNS["email"].sub(lambda m: _mask("email", m), text)
    out = PII_PATTERNS["ssn"].sub(lambda m: _mask("ssn", m), out)
    out = PII_PATTERNS["iban"].sub(lambda m: _mask("iban", m), out)

    def _card(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return (
            _mask("credit_card", m) if 13 <= len(digits) <= 16 and _luhn_ok(digits) else m.group(0)
        )

    out = PII_PATTERNS["credit_card"].sub(_card, out)

    def _phone(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return _mask("phone", m) if 9 <= len(digits) <= 13 else m.group(0)

    out = PII_PATTERNS["phone"].sub(_phone, out)
    return out, RedactionReport(counts=counts)


def redact_fields(fields: list[ExtractedField]) -> tuple[list[ExtractedField], RedactionReport]:
    """Redact PII inside string field values (identifier/amount fields are skipped)."""
    counts: dict[str, int] = {}
    out: list[ExtractedField] = []
    for f in fields:
        if f.name in SAFE_FIELDS or not isinstance(f.value, str):
            out.append(f)
            continue
        masked, report = redact_text(f.value)
        for k, v in report.counts.items():
            counts[k] = counts.get(k, 0) + v
        if report.total:
            out.append(f.model_copy(update={"value": masked, "redacted": True}))
        else:
            out.append(f)
    return out, RedactionReport(counts=counts)
