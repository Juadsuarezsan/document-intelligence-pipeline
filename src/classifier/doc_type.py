"""Document type classifier.

Three strategies, cheapest first:

1. Weighted keyword heuristic (always available, deterministic).
2. Claude over the first characters of text (when an API key exists).
3. Claude Vision over the first page image (for scans with no usable text).

Any failure to parse the model output is logged and falls back to the
heuristic with a lowered confidence; nothing is swallowed silently.
"""

from __future__ import annotations

import re
from typing import Any, Literal, get_args

from loguru import logger
from PIL import Image
from pydantic import BaseModel

from src.llm.client import ClaudeClient, LLMOutputError, image_block, text_block
from src.parsers.ocr_parser import image_to_png_bytes
from src.schemas.document import DocumentType, UsageStats

ClassifierMethod = Literal["heuristic", "claude", "claude_vision"]

DOCUMENT_TYPES: tuple[DocumentType, ...] = get_args(DocumentType)

# (pattern, weight). Patterns are matched case-insensitively on the whole text.
HEURISTIC_KEYWORDS: dict[DocumentType, list[tuple[str, float]]] = {
    "invoice": [
        (r"\binvoice\b", 3.0),
        (r"\bfactura\b", 3.0),
        (r"\bbill to\b", 2.0),
        (r"\b(?:nit|vat|tax id)\b", 1.5),
        (r"\bsubtotal\b", 1.5),
        (r"\bdue date\b", 1.0),
        (r"\bpayment terms\b", 1.0),
    ],
    "receipt": [
        (r"\breceipt\b", 3.0),
        (r"\brecibo\b", 3.0),
        (r"\bpayment received\b", 2.0),
        (r"\bcashier\b", 1.5),
        (r"\bchange due\b", 1.5),
        (r"\bthank you for (?:your|shopping)\b", 1.0),
        (r"\b(?:card|cash|visa|mastercard)\b", 0.5),
    ],
    "contract": [
        (r"\bagreement\b", 3.0),
        (r"\bcontract\b", 2.5),
        (r"\bparties\b", 1.5),
        (r"\beffective date\b", 1.5),
        (r"\bgoverning law\b", 2.0),
        (r"\bterm of\b", 1.0),
        (r"\bwhereas\b", 1.5),
        (r"\b(?:msa|sow)\b", 1.0),
    ],
    "form": [
        (r"\bform (?:id|no|number)\b", 3.0),
        (r"\bapplication form\b", 3.0),
        (r"\bsubmission\b", 1.0),
        (r"\bsubmitted by\b", 1.5),
        (r"\bplease (?:print|complete)\b", 1.5),
        (r"\bsignature\b", 1.0),
        (r"\bdepartment\b", 0.5),
        (r"\bdate:\s*", 0.25),
    ],
    "report": [
        (r"\bexecutive summary\b", 3.0),
        (r"\breport\b", 2.0),
        (r"\bfiscal year\b", 2.0),
        (r"\b10-k\b", 2.0),
        (r"\bprepared by\b", 1.5),
        (r"\bfindings\b", 1.0),
        (r"\btable of contents\b", 1.0),
        (r"\bauthor\b", 0.5),
    ],
}

CLASSIFY_SYSTEM = (
    "You classify business documents. Respond with ONLY a JSON object of the form "
    '{"type": "<invoice|receipt|contract|form|report>", "confidence": <0.0-1.0>}. '
    "Use 'form' for filled administrative forms with labelled fields."
)


class ClassificationResult(BaseModel):
    """Outcome of a classification call."""

    doc_type: DocumentType
    confidence: float
    method: ClassifierMethod
    scores: dict[str, float] = {}
    usage: UsageStats = UsageStats()


def heuristic_scores(text: str) -> dict[DocumentType, float]:
    """Score every document type from weighted keyword hits.

    Args:
        text: Document text.

    Returns:
        Mapping type -> score (0.0 when no keyword matched).
    """
    lowered = text.lower()
    scores: dict[DocumentType, float] = {}
    for doc_type, patterns in HEURISTIC_KEYWORDS.items():
        total = 0.0
        for pattern, weight in patterns:
            hits = len(re.findall(pattern, lowered))
            if hits:
                total += weight * min(hits, 3)
        scores[doc_type] = round(total, 3)
    return scores


def classify_heuristic(text: str) -> ClassificationResult:
    """Deterministic keyword classifier.

    Confidence is the margin between the best and second-best score, squashed
    to ``[0.35, 0.95]``; ``form`` is the default when nothing matches.
    """
    scores = heuristic_scores(text)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    if best_score <= 0:
        return ClassificationResult(
            doc_type="form", confidence=0.35, method="heuristic", scores=dict(scores)
        )
    margin = (best_score - second_score) / max(best_score, 1.0)
    confidence = 0.35 + 0.6 * min(1.0, margin)
    return ClassificationResult(
        doc_type=best, confidence=round(confidence, 3), method="heuristic", scores=dict(scores)
    )


def _parse_type(data: dict[str, Any]) -> tuple[DocumentType, float]:
    raw_type = str(data.get("type", "")).strip().lower()
    if raw_type not in DOCUMENT_TYPES:
        raise LLMOutputError(f"unknown document type from model: {raw_type!r}")
    doc_type: DocumentType = raw_type  # type: ignore[assignment]
    conf = data.get("confidence", 0.8)
    try:
        confidence = min(1.0, max(0.0, float(conf)))
    except (TypeError, ValueError) as exc:
        raise LLMOutputError("confidence is not numeric") from exc
    return doc_type, confidence


class DocTypeClassifier:
    """Classifier facade choosing between heuristic, Claude text and Claude Vision.

    Args:
        llm: Claude client; when ``None`` or disabled only the heuristic runs.
        max_chars: Characters of text sent to the model.
    """

    def __init__(self, llm: ClaudeClient | None = None, max_chars: int = 2000) -> None:
        self._llm = llm
        self._max_chars = max_chars

    @property
    def llm_enabled(self) -> bool:
        """True when a usable Claude client is attached."""
        return self._llm is not None and self._llm.enabled

    async def classify(self, text: str) -> ClassificationResult:
        """Classify document text.

        Uses Claude when available, otherwise (or on unparsable output) the
        heuristic. Model output problems are logged as warnings and the
        heuristic result is returned with ``confidence`` capped at 0.6.
        """
        if not self.llm_enabled or self._llm is None:
            return classify_heuristic(text)
        try:
            data, result = await self._llm.complete_json(
                system=CLASSIFY_SYSTEM, content=text[: self._max_chars], max_tokens=64
            )
            doc_type, confidence = _parse_type(data)
        except LLMOutputError as exc:
            logger.warning("classifier: unparsable model output ({}); using heuristic", exc)
            fallback = classify_heuristic(text)
            fallback.confidence = min(fallback.confidence, 0.6)
            return fallback
        return ClassificationResult(
            doc_type=doc_type, confidence=confidence, method="claude", usage=result.usage
        )

    async def classify_image(self, image: Image.Image) -> ClassificationResult:
        """Classify from a page image with Claude Vision.

        Raises:
            LLMDisabledError: If no API key is configured (propagated from the client).
            LLMOutputError: If the model output cannot be parsed.
        """
        if self._llm is None:
            from src.llm.client import LLMDisabledError

            raise LLMDisabledError("classifier has no Claude client")
        content = [
            image_block(image_to_png_bytes(image)),
            text_block("Classify this document page."),
        ]
        data, result = await self._llm.complete_json(
            system=CLASSIFY_SYSTEM, content=content, max_tokens=64
        )
        doc_type, confidence = _parse_type(data)
        return ClassificationResult(
            doc_type=doc_type, confidence=confidence, method="claude_vision", usage=result.usage
        )
