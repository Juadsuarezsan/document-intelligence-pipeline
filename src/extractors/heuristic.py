"""Deterministic field extractor (regex + layout key/value pairs).

This is the "Unstructured only" baseline of the evaluation: no model, no
network. It knows the label variants each schema field can appear under,
normalises values and assigns fixed confidences per rule strength.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.extractors.normalize import normalize_amount, normalize_date, normalize_text
from src.parsers.layout import Element, key_value_pairs, partition
from src.schemas.doc_types import DATE_FIELDS, NUMERIC_FIELDS
from src.schemas.document import DocumentType, ExtractedField

# Label variants per field, lower-case, matched against the key of a "Key: value" line.
LABELS: dict[str, list[str]] = {
    "invoice_number": [
        "invoice number",
        "invoice no",
        "invoice #",
        "invoice",
        "factura no",
        "factura",
        "inv no",
        "inv",
    ],
    "receipt_number": [
        "receipt number",
        "receipt no",
        "receipt #",
        "receipt",
        "recibo",
        "ticket",
        "transaction id",
        "txn",
    ],
    "contract_number": [
        "contract number",
        "contract no",
        "agreement number",
        "agreement no",
        "contract id",
        "ref",
    ],
    "form_id": ["form id", "form no", "form number", "form", "reference", "ref no", "case no"],
    "vendor": ["vendor", "seller", "from", "supplier", "issued by", "merchant", "store", "company"],
    "vendor_tax_id": ["nit", "tax id", "vat", "vat number", "vat id", "tin", "rfc", "cuit", "ein"],
    "issue_date": [
        "issue date",
        "invoice date",
        "date of issue",
        "date",
        "fecha",
        "issued",
        "date issued",
    ],
    "due_date": ["due date", "payment due", "due", "vencimiento"],
    "effective_date": ["effective date", "effective", "commencement date", "start date", "dated"],
    "date": ["date", "fecha", "submission date", "date submitted", "report date", "published"],
    "currency": ["currency", "moneda", "ccy"],
    "subtotal": ["subtotal", "sub total", "net amount", "net"],
    "tax": ["tax", "vat", "iva", "sales tax", "tax amount", "gst"],
    "total": [
        "total",
        "total due",
        "amount due",
        "grand total",
        "total amount",
        "balance due",
        "amount",
        "total paid",
    ],
    "total_value": [
        "total value",
        "contract value",
        "total contract value",
        "consideration",
        "fee",
        "total fees",
    ],
    "payment_terms": ["payment terms", "terms", "net"],
    "payment_method": ["payment method", "paid by", "payment", "tender", "method"],
    "parties": ["parties", "between", "party a", "client", "customer"],
    "term_months": ["term", "term months", "duration", "initial term"],
    "governing_law": ["governing law", "jurisdiction", "law"],
    "submitter": [
        "submitter",
        "submitted by",
        "applicant",
        "name",
        "employee",
        "requested by",
        "prepared by",
    ],
    "department": ["department", "dept", "division", "unit"],
    "title": ["title", "subject", "re"],
    "author": ["author", "prepared by", "written by", "by"],
    "fiscal_year": ["fiscal year", "fy", "year", "period"],
}

FIELDS_BY_TYPE: dict[DocumentType, list[str]] = {
    "invoice": [
        "invoice_number",
        "vendor",
        "vendor_tax_id",
        "issue_date",
        "due_date",
        "currency",
        "subtotal",
        "tax",
        "total",
        "payment_terms",
    ],
    "receipt": ["receipt_number", "vendor", "issue_date", "total", "currency", "payment_method"],
    "contract": [
        "contract_number",
        "parties",
        "effective_date",
        "term_months",
        "total_value",
        "currency",
        "governing_law",
    ],
    "form": ["form_id", "submitter", "date", "department"],
    "report": ["title", "author", "date", "fiscal_year"],
}

CURRENCY_RE = re.compile(r"\b(USD|EUR|COP|GBP|MXN|CAD|BRL|CLP|PEN|ARS|JPY|CHF)\b")
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
TERM_RE = re.compile(r"(\d{1,3})\s*(months?|meses|years?|años?)", re.IGNORECASE)
PAYMENT_METHODS = (
    "cash",
    "card",
    "visa",
    "mastercard",
    "amex",
    "transfer",
    "debit",
    "credit",
    "efectivo",
    "tarjeta",
)

_ID_LIKE = re.compile(r"[A-Z]{0,5}[-/ ]?\d{2,}[A-Z0-9\-/]*", re.IGNORECASE)


@dataclass
class ExtractionResult:
    """Fields produced by an extractor plus how they were produced."""

    fields: list[ExtractedField]
    method: str


def _kv_index(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for key, value in pairs:
        index.setdefault(normalize_text(key), []).append(value.strip())
    return index


def _lookup(index: dict[str, list[str]], field: str) -> tuple[str, float] | None:
    """Return the first value whose key matches one of the field's label variants."""
    for rank, label in enumerate(LABELS.get(field, [])):
        norm = normalize_text(label)
        if norm in index:
            return index[norm][0], max(0.55, 0.92 - 0.04 * rank)
        # Prefix match: "invoice no." vs "invoice no"
        for key, values in index.items():
            if key.startswith(norm + " ") or key.startswith(norm):
                if len(key) - len(norm) <= 12:
                    return values[0], max(0.5, 0.82 - 0.04 * rank)
    return None


def _coerce(field: str, raw: str) -> tuple[str | float | int | None, float]:
    """Normalise a raw value according to the field kind; returns (value, penalty)."""
    if field in NUMERIC_FIELDS:
        amount = normalize_amount(raw)
        return (amount, 0.0 if amount is not None else 0.4)
    if field == "term_months":
        m = TERM_RE.search(raw)
        if not m:
            return (None, 0.4)
        n = int(m.group(1))
        return (n * 12 if m.group(2).lower().startswith(("year", "año")) else n, 0.0)
    if field in DATE_FIELDS:
        iso = normalize_date(raw)
        return (iso if iso else raw, 0.0 if iso else 0.25)
    if field == "currency":
        m = CURRENCY_RE.search(raw.upper())
        return (m.group(1) if m else raw.strip()[:3].upper(), 0.0 if m else 0.3)
    if field == "payment_method":
        low = raw.lower()
        for method in PAYMENT_METHODS:
            if method in low:
                return (method, 0.0)
        return (raw.strip(), 0.2)
    if field in {"invoice_number", "receipt_number", "contract_number", "form_id"}:
        m = _ID_LIKE.search(raw)
        return (m.group(0).strip() if m else raw.strip(), 0.0 if m else 0.2)
    return (raw.strip(), 0.0)


def _currency_from_text(text: str) -> str | None:
    m = CURRENCY_RE.search(text)
    if m:
        return m.group(1)
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    return None


def _vendor_from_layout(elements: list[Element]) -> str | None:
    """First title-like line is usually the issuing company on invoices/receipts."""
    for el in elements[:6]:
        if el.category == "Title" and not re.search(
            r"invoice|receipt|factura|recibo|report|agreement|contract|form", el.text, re.IGNORECASE
        ):
            return el.text.strip()
    return None


def _parties_from_text(text: str) -> str | None:
    m = re.search(
        r"between\s+(.+?)\s+(?:\(.*?\)\s+)?and\s+(.+?)(?:\s*[\.,\n(]|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        a, b = (re.sub(r"\s+", " ", g).strip() for g in m.groups())
        return f"{a}; {b}"
    return None


def _title_from_layout(elements: list[Element]) -> str | None:
    for el in elements[:5]:
        if el.category == "Title":
            return el.text.strip()
    return None


class HeuristicExtractor:
    """Regex/layout extractor for all five document types."""

    def __init__(self, use_unstructured: bool = False) -> None:
        self._use_unstructured = use_unstructured

    def extract(self, text: str, doc_type: DocumentType) -> ExtractionResult:
        """Extract schema fields from text without any model call.

        Args:
            text: Full document text.
            doc_type: Schema to fill.

        Returns:
            Fields (only those found) with ``source='regex'``.
        """
        elements = partition(text, use_unstructured=self._use_unstructured)
        index = _kv_index(key_value_pairs(elements))
        fields: list[ExtractedField] = []
        seen: set[str] = set()

        def add(name: str, value: str | float | int | None, conf: float) -> None:
            if name in seen or value in (None, ""):
                return
            seen.add(name)
            fields.append(
                ExtractedField(
                    name=name,
                    value=value,
                    confidence=round(min(0.98, max(0.05, conf)), 3),
                    source="regex",
                )
            )

        for field in FIELDS_BY_TYPE[doc_type]:
            hit = _lookup(index, field)
            if hit is None:
                continue
            value, penalty = _coerce(field, hit[0])
            add(field, value, hit[1] - penalty)

        # Second-pass rules for fields that rarely appear as "Key: value".
        if doc_type in ("invoice", "receipt") and "currency" not in seen:
            cur = _currency_from_text(text)
            if cur:
                add("currency", cur, 0.7)
        if doc_type in ("invoice", "receipt") and "vendor" not in seen:
            vendor = _vendor_from_layout(elements)
            if vendor:
                add("vendor", vendor, 0.6)
        if doc_type == "contract" and "parties" not in seen:
            parties = _parties_from_text(text)
            if parties:
                add("parties", parties, 0.7)
        if doc_type == "contract" and "currency" not in seen:
            cur = _currency_from_text(text)
            if cur:
                add("currency", cur, 0.65)
        if doc_type == "report" and "title" not in seen:
            title = _title_from_layout(elements)
            if title:
                add("title", title, 0.6)
        if doc_type in ("invoice", "receipt") and "total" not in seen:
            m = re.search(r"total[^\d\n]{0,25}([\d.,]+\d)", text, re.IGNORECASE)
            if m:
                add("total", normalize_amount(m.group(1)), 0.6)
        if doc_type == "receipt" and "payment_method" not in seen:
            low = text.lower()
            for method in PAYMENT_METHODS:
                if re.search(rf"\b{method}\b", low):
                    add("payment_method", method, 0.6)
                    break
        return ExtractionResult(fields=fields, method="heuristic")

    def extract_pairs(self, text: str) -> list[tuple[str, str]]:
        """Return raw ``(key, value)`` pairs from the layout (FUNSD-style evaluation)."""
        return key_value_pairs(partition(text, use_unstructured=self._use_unstructured))
