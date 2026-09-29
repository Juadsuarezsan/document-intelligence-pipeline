"""Core domain models: parsed documents, extracted fields/tables, findings, results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DocumentType = Literal["invoice", "contract", "form", "report", "receipt"]
"""Supported document classes (``auto`` is an API-level request value, not a type)."""

Routing = Literal["auto_approve", "human_review", "vlm_fallback", "rejected"]
"""Decision taken by the confidence scorer."""

InputPath = Literal["text", "ocr", "vlm"]
"""Parsing route actually used for a document."""

FieldSource = Literal["regex", "extractor", "vlm", "human"]
"""Which component produced a field value."""

Severity = Literal["error", "warn", "info"]

FieldValue = str | float | int | bool | None


class ExtractedField(BaseModel):
    """A single named value pulled out of a document."""

    name: str = Field(..., min_length=1, description="Schema field name, e.g. ``total``")
    value: FieldValue = Field(default=None, description="Normalised value or ``None``")
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: FieldSource = "extractor"
    page: int | None = Field(default=None, ge=1)
    bbox: list[float] | None = Field(
        default=None, description="[x0, y0, x1, y1] normalised to the page", min_length=4
    )
    redacted: bool = Field(default=False, description="True when PII redaction masked the value")


class ExtractedTable(BaseModel):
    """A table recovered from a page as header row plus body rows."""

    page: int = Field(..., ge=1)
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    backend: Literal["pdfplumber", "camelot", "vlm"] = "pdfplumber"

    @property
    def n_cells(self) -> int:
        """Total number of cells including the header row."""
        return len(self.headers) + sum(len(r) for r in self.rows)


class Finding(BaseModel):
    """A validation outcome attached to a document."""

    severity: Severity
    code: str
    field: str | None = None
    message: str = ""
    details: dict[str, str | float | int | None] = Field(default_factory=dict)


class PageText(BaseModel):
    """Text of one page with the confidence the parser assigns to it."""

    number: int = Field(..., ge=1)
    text: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    has_tables: bool = False


class ParsedDocument(BaseModel):
    """Output of any parser (text, OCR or Vision)."""

    pages: list[PageText] = Field(default_factory=list)
    source: Literal["text", "ocr", "vlm", "unstructured", "raw"] = "text"

    @property
    def text(self) -> str:
        """Full document text with pages separated by blank lines."""
        return "\n\n".join(p.text for p in self.pages)

    @property
    def n_pages(self) -> int:
        """Number of pages parsed."""
        return len(self.pages)

    @property
    def has_tables(self) -> bool:
        """Whether any page reported a table candidate."""
        return any(p.has_tables for p in self.pages)

    @property
    def chars_per_page(self) -> float:
        """Mean extractable characters per page (0.0 for an empty document)."""
        if not self.pages:
            return 0.0
        return sum(len(p.text.strip()) for p in self.pages) / len(self.pages)

    @property
    def mean_confidence(self) -> float:
        """Mean parser confidence across pages (1.0 for an empty document)."""
        if not self.pages:
            return 1.0
        return sum(p.confidence for p in self.pages) / len(self.pages)

    def is_scanned(self, min_chars_per_page: int) -> bool:
        """Return True when the text layer is too thin to trust (likely a scan)."""
        return self.n_pages > 0 and self.chars_per_page < min_chars_per_page


class UsageStats(BaseModel):
    """Token and cost accounting for one request (summed over all LLM calls)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0

    def add(self, other: UsageStats) -> UsageStats:
        """Return a new object with both usages summed."""
        return UsageStats(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
            llm_calls=self.llm_calls + other.llm_calls,
        )


class PipelineResult(BaseModel):
    """Everything the pipeline produced for one document."""

    trace_id: str
    document_type: DocumentType
    classifier_method: Literal["requested", "heuristic", "claude", "claude_vision"]
    fields: list[ExtractedField] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    routing: Routing
    pipeline_used: InputPath = "text"
    extractor_method: Literal["heuristic", "claude", "claude_vision"] = "heuristic"
    vlm_reprocessed: bool = False
    n_pages: int = 0
    latency_ms: int = 0
    usage: UsageStats = Field(default_factory=UsageStats)
    node_log: list[dict[str, str | int | float]] = Field(default_factory=list)
