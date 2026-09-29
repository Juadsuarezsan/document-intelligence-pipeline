"""Claude-based field extractor with schema-constrained JSON output.

Without an API key the extractor delegates to :class:`HeuristicExtractor`.
With a key, model output is validated into :class:`ExtractedField`; malformed
output is logged, reported as a finding-worthy ``method`` and the heuristic
result is returned instead (never a silent pass).
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from PIL import Image
from pydantic import ValidationError

from src.extractors.heuristic import HeuristicExtractor
from src.extractors.normalize import normalize_amount, normalize_date
from src.llm.client import ClaudeClient, LLMOutputError, image_block, text_block
from src.parsers.ocr_parser import image_to_png_bytes
from src.schemas.doc_types import DATE_FIELDS, INTEGER_FIELDS, NUMERIC_FIELDS, schema_field_names
from src.schemas.document import DocumentType, ExtractedField, FieldSource, UsageStats

SYSTEM_TEMPLATE = """You extract structured fields from {doc_type} documents.

Return ONLY a JSON object with this shape:
{{"fields": [{{"name": "<field>", "value": <value or null>, "confidence": <0.0-1.0>}}, ...]}}

Fields to extract for {doc_type} (use exactly these names):
{field_list}

Rules:
1. Output one entry per field above; use null when the value is absent.
2. Numbers as plain numbers (no currency symbols, no thousands separators).
3. Dates in ISO-8601 (YYYY-MM-DD).
4. Identifiers as strings exactly as written.
5. Confidence: 0.95+ verbatim in source; 0.80-0.95 derived (e.g. subtotal+tax=total);
   0.60-0.80 ambiguous; below 0.60 prefer null.
"""


class LLMExtractionResult:
    """Fields plus provenance and usage for one extraction call."""

    def __init__(self, fields: list[ExtractedField], method: str, usage: UsageStats) -> None:
        self.fields = fields
        self.method = method
        self.usage = usage


def _system_prompt(doc_type: DocumentType) -> str:
    names = [n for n in schema_field_names(doc_type) if n != "fields"]
    return SYSTEM_TEMPLATE.format(
        doc_type=doc_type, field_list="\n".join(f"  - {n}" for n in names)
    )


def fields_from_model_output(
    data: dict[str, Any], doc_type: DocumentType, source: FieldSource
) -> list[ExtractedField]:
    """Validate the model's ``{"fields": [...]}`` payload into domain objects.

    Unknown field names are dropped, values are coerced by field kind and
    ``null`` values are skipped.

    Raises:
        LLMOutputError: If the payload has the wrong shape or invalid entries.
    """
    raw_fields = data.get("fields")
    if not isinstance(raw_fields, list):
        raise LLMOutputError("model output missing 'fields' list")
    allowed = set(schema_field_names(doc_type))
    out: list[ExtractedField] = []
    for item in raw_fields:
        if not isinstance(item, dict):
            raise LLMOutputError("field entry is not an object")
        name = str(item.get("name", ""))
        if name not in allowed:
            logger.debug("extractor: dropping unknown field {!r}", name)
            continue
        value = item.get("value")
        if value is None or value == "":
            continue
        if name in NUMERIC_FIELDS:
            value = normalize_amount(value if isinstance(value, int | float) else str(value))
        elif name in INTEGER_FIELDS:
            try:
                value = int(float(str(value)))
            except ValueError as exc:
                raise LLMOutputError(f"{name} is not an integer") from exc
        elif name in DATE_FIELDS:
            value = normalize_date(str(value)) or str(value)
        elif isinstance(value, dict | list):
            value = str(value)
        try:
            out.append(
                ExtractedField(
                    name=name,
                    value=value,
                    confidence=float(item.get("confidence", 0.7)),
                    source=source,
                )
            )
        except (ValidationError, TypeError, ValueError) as exc:
            raise LLMOutputError(f"invalid field entry {name!r}: {exc}") from exc
    return out


class LLMExtractor:
    """Text extractor: Claude when enabled, heuristic otherwise.

    Args:
        llm: Claude client (may be ``None``).
        heuristic: Fallback extractor.
        max_chars: Characters of document text sent to the model.
    """

    def __init__(
        self,
        llm: ClaudeClient | None,
        heuristic: HeuristicExtractor | None = None,
        max_chars: int = 12_000,
    ) -> None:
        self._llm = llm
        self._heuristic = heuristic or HeuristicExtractor()
        self._max_chars = max_chars

    @property
    def llm_enabled(self) -> bool:
        """True when Claude will be used."""
        return self._llm is not None and self._llm.enabled

    async def extract(self, text: str, doc_type: DocumentType) -> LLMExtractionResult:
        """Extract fields from text.

        Returns:
            Result whose ``method`` is ``claude``, ``heuristic`` or
            ``heuristic_after_llm_error`` (Claude answered but unparsably).
        """
        if not self.llm_enabled or self._llm is None:
            res = self._heuristic.extract(text, doc_type)
            return LLMExtractionResult(res.fields, "heuristic", UsageStats())
        try:
            data, result = await self._llm.complete_json(
                system=_system_prompt(doc_type), content=text[: self._max_chars], max_tokens=1500
            )
            fields = fields_from_model_output(data, doc_type, "extractor")
        except LLMOutputError as exc:
            logger.warning("extractor: unusable model output ({}); using heuristic", exc)
            res = self._heuristic.extract(text, doc_type)
            return LLMExtractionResult(res.fields, "heuristic_after_llm_error", UsageStats())
        return LLMExtractionResult(fields, "claude", result.usage)


class VLMExtractor:
    """Claude Vision extractor working directly on page images ("Vision direct")."""

    def __init__(self, llm: ClaudeClient, max_pages: int = 4) -> None:
        self._llm = llm
        self._max_pages = max_pages

    @property
    def enabled(self) -> bool:
        """True when the client has credentials."""
        return self._llm.enabled

    async def extract(
        self, images: list[Image.Image], doc_type: DocumentType
    ) -> LLMExtractionResult:
        """Extract fields from page images.

        Raises:
            LLMDisabledError: If no API key is configured.
            LLMOutputError: If the model output cannot be validated.
        """
        content: list[dict[str, Any]] = [
            image_block(image_to_png_bytes(img)) for img in images[: self._max_pages]
        ]
        content.append(text_block(f"Extract the {doc_type} fields from these pages."))
        data, result = await self._llm.complete_json(
            system=_system_prompt(doc_type), content=content, max_tokens=1500
        )
        fields = fields_from_model_output(data, doc_type, "vlm")
        return LLMExtractionResult(fields, "claude_vision", result.usage)
