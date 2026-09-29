"""HTTP request/response models (transport layer only).

Domain models live in :mod:`src.schemas`; these classes only shape what goes
over the wire and enforce request validation so bad input returns 422.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from src.schemas.document import (
    DocumentType,
    ExtractedField,
    ExtractedTable,
    Finding,
    InputPath,
    PipelineResult,
    Routing,
    UsageStats,
)

RequestedType = Literal["auto", "invoice", "contract", "form", "report", "receipt"]


class ExtractRequest(BaseModel):
    """Body of ``POST /api/extract``: exactly one of ``text`` or ``pdf_b64``."""

    document_type: RequestedType = Field(default="auto", description="Skip classification")
    text: str | None = Field(default=None, description="Pre-extracted document text")
    pdf_b64: str | None = Field(default=None, description="Base64-encoded PDF")
    force_vlm: bool = Field(default=False, description="Route straight to Claude Vision")

    @model_validator(mode="after")
    def _exactly_one_input(self) -> Self:
        has_text = bool(self.text and self.text.strip())
        has_pdf = bool(self.pdf_b64 and self.pdf_b64.strip())
        if has_text == has_pdf:
            raise ValueError("provide exactly one of 'text' or 'pdf_b64' (non-empty)")
        return self

    @property
    def requested_type(self) -> DocumentType | None:
        """``None`` when the caller asked for automatic classification."""
        return None if self.document_type == "auto" else self.document_type


class ExtractResponse(BaseModel):
    """Result of an extraction request."""

    trace_id: str
    document_type: DocumentType
    classifier_method: str
    fields: list[ExtractedField] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    overall_confidence: float = 0.0
    routing: Routing
    validation_findings: list[Finding] = Field(default_factory=list)
    pipeline_used: InputPath = "text"
    extractor_method: str = "heuristic"
    vlm_reprocessed: bool = False
    n_pages: int = 0
    latency_ms: int = 0
    usage: UsageStats = Field(default_factory=UsageStats)
    node_log: list[dict[str, str | int | float]] = Field(default_factory=list)

    @classmethod
    def from_result(cls, result: PipelineResult) -> ExtractResponse:
        """Map a domain result onto the wire format."""
        return cls(
            trace_id=result.trace_id,
            document_type=result.document_type,
            classifier_method=result.classifier_method,
            fields=result.fields,
            tables=result.tables,
            overall_confidence=result.overall_confidence,
            routing=result.routing,
            validation_findings=result.findings,
            pipeline_used=result.pipeline_used,
            extractor_method=result.extractor_method,
            vlm_reprocessed=result.vlm_reprocessed,
            n_pages=result.n_pages,
            latency_ms=result.latency_ms,
            usage=result.usage,
            node_log=result.node_log,
        )


class HealthResponse(BaseModel):
    """``GET /health`` payload."""

    status: Literal["ok"]
    version: str
    model: str
    llm_enabled: bool
    vlm_fallback: bool
    ocr_available: bool
    database: bool


class MetricsResponse(BaseModel):
    """``GET /api/metrics/recent`` payload: last requests plus aggregates."""

    count: int
    p50_ms: int
    p95_ms: int
    total_cost_usd: float
    requests: list[dict[str, str | int | float]]
