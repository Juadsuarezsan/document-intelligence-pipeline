"""Domain models shared by parsers, extractors, validators, pipeline and API.

Nothing in this package imports FastAPI or any I/O library: the HTTP layer
(``src/api``) depends on these models, never the other way around.
"""

from src.schemas.doc_types import (
    CRITICAL_FIELDS,
    DOCUMENT_SCHEMAS,
    REQUIRED_FIELDS,
    ContractSchema,
    FormSchema,
    InvoiceSchema,
    ReceiptSchema,
    ReportSchema,
)
from src.schemas.document import (
    DocumentType,
    ExtractedField,
    ExtractedTable,
    FieldSource,
    Finding,
    InputPath,
    PageText,
    ParsedDocument,
    PipelineResult,
    Routing,
    Severity,
    UsageStats,
)

__all__ = [
    "CRITICAL_FIELDS",
    "DOCUMENT_SCHEMAS",
    "REQUIRED_FIELDS",
    "ContractSchema",
    "DocumentType",
    "ExtractedField",
    "ExtractedTable",
    "FieldSource",
    "Finding",
    "FormSchema",
    "InputPath",
    "InvoiceSchema",
    "PageText",
    "ParsedDocument",
    "PipelineResult",
    "ReceiptSchema",
    "ReportSchema",
    "Routing",
    "Severity",
    "UsageStats",
]
