"""Pydantic schema per document type.

Each schema documents the fields the extractor must produce. ``REQUIRED_FIELDS``
drives the completeness validator and ``CRITICAL_FIELDS`` defines document-level
accuracy: a document counts as correct only when every critical field matches
the ground truth.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.schemas.document import DocumentType


class InvoiceSchema(BaseModel):
    """Commercial invoice."""

    invoice_number: str | None = Field(default=None, description="Vendor's invoice identifier")
    vendor: str | None = Field(default=None, description="Issuing company name")
    vendor_tax_id: str | None = Field(default=None, description="NIT / VAT / Tax ID")
    issue_date: str | None = Field(default=None, description="ISO-8601 date")
    due_date: str | None = Field(default=None, description="ISO-8601 date")
    currency: str | None = Field(default=None, description="ISO-4217 code")
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    payment_terms: str | None = Field(default=None, description="e.g. ``Net 30``")


class ReceiptSchema(BaseModel):
    """Point-of-sale receipt."""

    receipt_number: str | None = None
    vendor: str | None = None
    issue_date: str | None = None
    total: float | None = None
    currency: str | None = None
    payment_method: str | None = Field(default=None, description="cash | card | transfer")


class ContractSchema(BaseModel):
    """Services or supply agreement."""

    contract_number: str | None = None
    parties: str | None = Field(default=None, description="``A; B`` joined with ``;``")
    effective_date: str | None = None
    term_months: int | None = None
    total_value: float | None = None
    currency: str | None = None
    governing_law: str | None = None


class FormSchema(BaseModel):
    """Generic administrative form (FUNSD-like key/value pairs)."""

    form_id: str | None = None
    submitter: str | None = None
    date: str | None = None
    department: str | None = None
    fields: dict[str, str] = Field(
        default_factory=dict, description="Free key/value pairs found in the form"
    )


class ReportSchema(BaseModel):
    """Narrative report (annual, audit, technical)."""

    title: str | None = None
    author: str | None = None
    date: str | None = None
    fiscal_year: str | None = None
    summary: str | None = None


DOCUMENT_SCHEMAS: dict[DocumentType, type[BaseModel]] = {
    "invoice": InvoiceSchema,
    "receipt": ReceiptSchema,
    "contract": ContractSchema,
    "form": FormSchema,
    "report": ReportSchema,
}

REQUIRED_FIELDS: dict[DocumentType, list[str]] = {
    "invoice": [
        "invoice_number",
        "vendor",
        "vendor_tax_id",
        "issue_date",
        "currency",
        "subtotal",
        "tax",
        "total",
    ],
    "receipt": ["receipt_number", "vendor", "issue_date", "total", "currency", "payment_method"],
    "contract": ["contract_number", "parties", "effective_date", "term_months", "total_value"],
    "form": ["form_id", "submitter", "date"],
    "report": ["title", "author", "date"],
}

CRITICAL_FIELDS: dict[DocumentType, list[str]] = {
    "invoice": ["invoice_number", "total", "issue_date"],
    "receipt": ["receipt_number", "total"],
    "contract": ["contract_number", "effective_date", "total_value"],
    "form": ["form_id", "date"],
    "report": ["title", "date"],
}

NUMERIC_FIELDS: frozenset[str] = frozenset({"subtotal", "tax", "total", "total_value"})
INTEGER_FIELDS: frozenset[str] = frozenset({"term_months"})
DATE_FIELDS: frozenset[str] = frozenset({"issue_date", "due_date", "effective_date", "date"})


def schema_field_names(doc_type: DocumentType) -> list[str]:
    """Return the declared field names for a document type.

    Args:
        doc_type: One of the supported document types.

    Returns:
        Field names in declaration order.
    """
    return list(DOCUMENT_SCHEMAS[doc_type].model_fields)
