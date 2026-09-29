"""Claude Vision parsing for documents whose text layer and OCR are both unreliable."""

from __future__ import annotations

from typing import Any

from loguru import logger
from PIL import Image

from src.llm.client import ClaudeClient, LLMOutputError, image_block, text_block
from src.parsers.ocr_parser import image_to_png_bytes
from src.schemas.document import PageText, ParsedDocument, UsageStats

TRANSCRIBE_SYSTEM = """You transcribe scanned business documents.

Rules:
1. Return ONLY a JSON object: {"pages": [{"text": "...", "confidence": 0.0-1.0}]} with one
   entry per image, in order.
2. Preserve reading order. Keep each form field on its own line as "Label: value".
3. Do not invent text. If a region is illegible write "[illegible]".
4. confidence is your estimate of transcription accuracy for the page.
"""


class VLMParser:
    """Transcribe page images with Claude Vision.

    Args:
        llm: Configured Claude client. When ``llm.enabled`` is False the parser
            raises :class:`LLMDisabledError` from the client on use.
    """

    def __init__(self, llm: ClaudeClient) -> None:
        self._llm = llm

    @property
    def enabled(self) -> bool:
        """Whether the underlying client has credentials."""
        return self._llm.enabled

    async def transcribe(self, images: list[Image.Image]) -> tuple[ParsedDocument, UsageStats]:
        """Transcribe page images into a :class:`ParsedDocument`.

        Args:
            images: Page renders, in order.

        Returns:
            The parsed document (``source='vlm'``) and the token usage.

        Raises:
            LLMOutputError: If the model does not return the expected JSON.
        """
        content: list[dict[str, Any]] = [image_block(image_to_png_bytes(img)) for img in images]
        content.append(text_block(f"Transcribe these {len(images)} page(s)."))
        data, result = await self._llm.complete_json(
            system=TRANSCRIBE_SYSTEM, content=content, max_tokens=4096
        )
        raw_pages = data.get("pages")
        if not isinstance(raw_pages, list) or not raw_pages:
            raise LLMOutputError("vision transcription missing 'pages'")
        pages: list[PageText] = []
        for i, item in enumerate(raw_pages, start=1):
            if not isinstance(item, dict):
                raise LLMOutputError("vision page entry is not an object")
            conf = float(item.get("confidence", 0.8))
            pages.append(
                PageText(
                    number=i,
                    text=str(item.get("text", "")),
                    confidence=min(1.0, max(0.0, conf)),
                )
            )
        logger.info("vlm transcribed pages={} tokens_in={}", len(pages), result.usage.input_tokens)
        return ParsedDocument(pages=pages, source="vlm"), result.usage
