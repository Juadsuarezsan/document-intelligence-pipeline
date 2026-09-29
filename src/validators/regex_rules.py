"""Format validators for identifiers, dates and amounts."""

from __future__ import annotations

import re
from datetime import date

from src.schemas.document import ExtractedField, Finding

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
NIT_RE = re.compile(r"^(\d{6,10})[-\s]?(\d)$")
GENERIC_TAX_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-/ ]{4,24}$", re.IGNORECASE)

_NIT_WEIGHTS = (3, 7, 13, 17, 19, 23, 29, 37, 41, 43, 47, 53, 59, 67, 71)


def nit_check_digit(base: str) -> int:
    """Compute the DIAN (Colombia) verification digit of a NIT.

    Args:
        base: NIT digits without the check digit.

    Returns:
        Check digit 0-9.
    """
    digits = [int(c) for c in base if c.isdigit()]
    total = sum(d * w for d, w in zip(reversed(digits), _NIT_WEIGHTS, strict=False))
    rem = total % 11
    return rem if rem < 2 else 11 - rem


def is_valid_nit(value: str) -> bool:
    """Validate ``NNNNNNNNN-D`` Colombian NITs (dots and spaces tolerated)."""
    cleaned = value.replace(".", "").replace(" ", "").strip()
    m = NIT_RE.match(cleaned)
    if not m:
        return False
    return nit_check_digit(m.group(1)) == int(m.group(2))


def check_tax_id(fields: list[ExtractedField]) -> list[Finding]:
    """Validate ``vendor_tax_id`` (NIT with check digit, or a plausible generic id)."""
    findings: list[Finding] = []
    for f in fields:
        if f.name != "vendor_tax_id" or not isinstance(f.value, str):
            continue
        cleaned = f.value.replace(".", "").replace(" ", "")
        if NIT_RE.match(cleaned):
            if not is_valid_nit(f.value):
                findings.append(
                    Finding(
                        severity="error",
                        code="invalid_nit_check_digit",
                        field=f.name,
                        message="NIT verification digit does not match",
                        details={"value": f.value},
                    )
                )
        elif not GENERIC_TAX_ID_RE.match(f.value):
            findings.append(
                Finding(
                    severity="warn",
                    code="implausible_tax_id",
                    field=f.name,
                    message="tax id has unexpected characters or length",
                    details={"value": f.value},
                )
            )
    return findings


def check_iso_dates(fields: list[ExtractedField]) -> list[Finding]:
    """Flag date fields that are not ISO-8601 or are not real calendar dates."""
    findings: list[Finding] = []
    for f in fields:
        if not (f.name.endswith("_date") or f.name == "date") or not isinstance(f.value, str):
            continue
        if not ISO_DATE_RE.match(f.value):
            findings.append(
                Finding(
                    severity="warn",
                    code="non_iso_date",
                    field=f.name,
                    message="date is not YYYY-MM-DD",
                    details={"value": f.value},
                )
            )
            continue
        try:
            date.fromisoformat(f.value)
        except ValueError:
            findings.append(
                Finding(
                    severity="error",
                    code="invalid_date",
                    field=f.name,
                    message="not a valid calendar date",
                    details={"value": f.value},
                )
            )
    return findings


def check_amounts(fields: list[ExtractedField]) -> list[Finding]:
    """Amounts must be numeric and non-negative; currency must be an ISO-4217 code."""
    findings: list[Finding] = []
    for f in fields:
        if f.name in {"subtotal", "tax", "total", "total_value"}:
            if not isinstance(f.value, int | float):
                findings.append(
                    Finding(
                        severity="error",
                        code="non_numeric_amount",
                        field=f.name,
                        message="amount is not numeric",
                        details={"value": str(f.value)},
                    )
                )
            elif f.value < 0:
                findings.append(
                    Finding(
                        severity="warn",
                        code="negative_amount",
                        field=f.name,
                        message="amount is negative",
                        details={"value": float(f.value)},
                    )
                )
        elif f.name == "currency" and isinstance(f.value, str) and not CURRENCY_RE.match(f.value):
            findings.append(
                Finding(
                    severity="warn",
                    code="invalid_currency_code",
                    field=f.name,
                    message="currency is not a 3-letter ISO code",
                    details={"value": f.value},
                )
            )
    return findings
