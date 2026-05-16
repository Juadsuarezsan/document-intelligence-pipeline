"""Schemas for Document Intelligence pipeline."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

DocumentType = Literal["invoice", "contract", "form", "report", "receipt", "auto"]
Routing = Literal["auto_approve", "human_review", "vlm_fallback", "rejected"]


class ExtractedField(BaseModel):
    name: str
    value: str | float | int | bool | None
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: Literal["regex", "extractor", "vlm", "human"] = "extractor"
    page: int | None = None
    bbox: list[float] | None = None  # [x0,y0,x1,y1] normalized


class ExtractedTable(BaseModel):
    page: int
    headers: list[str]
    rows: list[list[str]]
    confidence: float = Field(..., ge=0.0, le=1.0)


class ExtractRequest(BaseModel):
    document_type: DocumentType = "auto"
    text: str | None = Field(default=None, description="Raw document text (alternative to file)")
    pdf_b64: str | None = None
    file_url: str | None = None
    force_vlm: bool = False


class ExtractResponse(BaseModel):
    document_type: DocumentType
    fields: list[ExtractedField] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    overall_confidence: float = 0.0
    routing: Routing
    validation_findings: list[dict[str, Any]] = Field(default_factory=list)
    pipeline_used: str = "text"  # text | ocr | vlm
    latency_ms: int = 0
