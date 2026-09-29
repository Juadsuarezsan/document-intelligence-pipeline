"""Decide which parsing route a document takes: text layer, OCR or Claude Vision."""

from __future__ import annotations

from src.schemas.document import InputPath, ParsedDocument


def choose_input_path(
    parsed: ParsedDocument | None,
    *,
    force_vlm: bool,
    vlm_enabled: bool,
    ocr_available: bool,
    min_chars_per_page: int,
) -> InputPath:
    """Pick the parsing route.

    Args:
        parsed: Result of the cheap text-layer parse (``None`` if no PDF was given).
        force_vlm: Caller demanded Claude Vision.
        vlm_enabled: An API key is configured and the Vision fallback is on.
        ocr_available: The tesseract binary is installed.
        min_chars_per_page: Threshold below which a page counts as scanned.

    Returns:
        ``"vlm"`` when forced (and available), ``"ocr"`` for scans with a working
        OCR engine, ``"vlm"`` for scans without OCR but with Vision, otherwise
        ``"text"``.
    """
    if force_vlm and vlm_enabled:
        return "vlm"
    if parsed is None:
        return "text"
    if parsed.is_scanned(min_chars_per_page):
        if ocr_available:
            return "ocr"
        if vlm_enabled:
            return "vlm"
    return "text"
