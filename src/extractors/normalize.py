"""Value normalisation shared by extractors and the evaluation metrics."""

from __future__ import annotations

import re
from datetime import date, datetime

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january", "ene", "enero"),
            ("feb", "february", "febrero"),
            ("mar", "march", "marzo"),
            ("apr", "april", "abr", "abril"),
            ("may", "mayo"),
            ("jun", "june", "junio"),
            ("jul", "july", "julio"),
            ("aug", "august", "ago", "agosto"),
            ("sep", "sept", "september", "septiembre"),
            ("oct", "october", "octubre"),
            ("nov", "november", "noviembre"),
            ("dec", "december", "dic", "diciembre"),
        ],
        start=1,
    )
    for m in names
}

_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b")
_MDY_TEXT_RE = re.compile(r"\b([A-Za-z]{3,10})\.?\s+(\d{1,2}),?\s+(\d{4})\b")
_DMY_TEXT_RE = re.compile(r"\b(\d{1,2})\s+(?:de\s+)?([A-Za-z]{3,10})\.?,?\s+(?:de\s+)?(\d{4})\b")
_AMOUNT_RE = re.compile(r"-?\$?\s*(\d{1,3}(?:[.,]\d{3})*|\d+)(?:[.,](\d{1,2}))?")


def normalize_date(value: str, day_first: bool = True) -> str | None:
    """Convert common date spellings to ISO-8601.

    Handles ``2026-03-14``, ``14/03/2026`` (day first by default), ``March 14, 2026``
    and ``14 March 2026``. Returns ``None`` when nothing parses.

    Args:
        value: Raw text.
        day_first: Interpret ``a/b/yyyy`` as day/month/year.
    """
    text = value.strip()
    if m := _ISO_RE.search(text):
        y, mo, d = (int(g) for g in m.groups())
        return _safe_iso(y, mo, d)
    if m := _DMY_RE.search(text):
        a, b, y = (int(g) for g in m.groups())
        d, mo = (a, b) if day_first else (b, a)
        if mo > 12 and d <= 12:
            d, mo = mo, d
        return _safe_iso(y, mo, d)
    if m := _MDY_TEXT_RE.search(text):
        month = _MONTHS.get(m.group(1).lower())
        if month is not None:
            return _safe_iso(int(m.group(3)), month, int(m.group(2)))
    if m := _DMY_TEXT_RE.search(text):
        month = _MONTHS.get(m.group(2).lower())
        if month is not None:
            return _safe_iso(int(m.group(3)), month, int(m.group(1)))
    return None


def _safe_iso(y: int, mo: int, d: int) -> str | None:
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


def normalize_amount(value: str | float | int) -> float | None:
    """Parse a money amount written with either ``1,234.56`` or ``1.234,56`` conventions.

    Returns:
        Float or ``None`` if no number is present.
    """
    if isinstance(value, int | float):
        return float(value)
    text = value.replace(" ", " ").strip()
    m = _AMOUNT_RE.search(text)
    if not m:
        return None
    integer_part, decimals = m.group(1), m.group(2)
    digits = re.sub(r"[.,]", "", integer_part)
    if not digits:
        return None
    number = float(digits)
    if decimals:
        number += float(decimals) / (10 ** len(decimals))
    if text.lstrip().startswith("-"):
        number = -number
    return round(number, 2)


def normalize_text(value: str) -> str:
    """Lower-case, collapse whitespace and strip surrounding punctuation."""
    text = re.sub(r"\s+", " ", value.strip().lower())
    return text.strip(" .,:;-_\"'()")


def today_iso() -> str:
    """Return today's date in ISO format (kept here so tests can patch it)."""
    return datetime.now().date().isoformat()
