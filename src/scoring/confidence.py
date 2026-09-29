"""Per-field and document-level confidence, then routing.

Document confidence = mean field confidence, weighted by the fraction of
required fields that were found, minus a penalty per validation error. The
routing thresholds come from settings (auto approve >= 0.90, human review >=
0.70, else re-process with the Vision model).
"""

from __future__ import annotations

from src.schemas.doc_types import REQUIRED_FIELDS
from src.schemas.document import DocumentType, ExtractedField, Finding, Routing

ERROR_PENALTY = 0.10
WARN_PENALTY = 0.03


def mean_field_confidence(fields: list[ExtractedField]) -> float:
    """Mean of field confidences (0.0 when no fields)."""
    if not fields:
        return 0.0
    return sum(f.confidence for f in fields) / len(fields)


def coverage(fields: list[ExtractedField], doc_type: DocumentType) -> float:
    """Fraction of required fields that have a non-empty value."""
    required = REQUIRED_FIELDS.get(doc_type, [])
    if not required:
        return 1.0
    have = {f.name for f in fields if f.value not in (None, "")}
    return sum(1 for r in required if r in have) / len(required)


def penalize_fields(fields: list[ExtractedField], findings: list[Finding]) -> list[ExtractedField]:
    """Lower the confidence of fields named in error findings (returns new objects)."""
    flagged = {f.field for f in findings if f.severity == "error" and f.field}
    out: list[ExtractedField] = []
    for f in fields:
        if f.name in flagged:
            out.append(f.model_copy(update={"confidence": round(max(0.0, f.confidence - 0.3), 3)}))
        else:
            out.append(f)
    return out


def document_confidence(
    fields: list[ExtractedField], findings: list[Finding], doc_type: DocumentType
) -> float:
    """Compute the global confidence in ``[0, 1]``."""
    base = mean_field_confidence(fields) * (0.5 + 0.5 * coverage(fields, doc_type))
    n_err = sum(1 for f in findings if f.severity == "error")
    n_warn = sum(1 for f in findings if f.severity == "warn")
    return round(max(0.0, min(1.0, base - ERROR_PENALTY * n_err - WARN_PENALTY * n_warn)), 3)


def routing_decision(confidence: float, *, auto_thr: float, review_thr: float) -> Routing:
    """Map confidence to a routing decision.

    Args:
        confidence: Document confidence.
        auto_thr: Threshold for automatic approval.
        review_thr: Threshold for the human review queue.
    """
    if confidence >= auto_thr:
        return "auto_approve"
    if confidence >= review_thr:
        return "human_review"
    return "vlm_fallback"
