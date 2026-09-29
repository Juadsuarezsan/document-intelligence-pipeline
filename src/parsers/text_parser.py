"""Text-layer PDF parsing with pdfplumber plus plain-text ingestion."""

from __future__ import annotations

import base64
import binascii
import io

from loguru import logger

from src.parsers.errors import EmptyInputError, InvalidPDFError, OversizedInputError
from src.schemas.document import PageText, ParsedDocument

PDF_MAGIC = b"%PDF"


def decode_pdf_b64(pdf_b64: str, max_bytes: int) -> bytes:
    """Decode and sanity-check a base64 PDF payload.

    Args:
        pdf_b64: Base64 string (data-URL prefixes are tolerated).
        max_bytes: Maximum accepted decoded size.

    Returns:
        Raw PDF bytes.

    Raises:
        InvalidPDFError: If the string is not base64 or does not start with ``%PDF``.
        OversizedInputError: If the decoded payload exceeds ``max_bytes``.
        EmptyInputError: If the payload is empty.
    """
    if not pdf_b64 or not pdf_b64.strip():
        raise EmptyInputError("pdf_b64 is empty")
    payload = pdf_b64.split(",", 1)[1] if pdf_b64.startswith("data:") else pdf_b64
    # base64 expands 3 bytes to 4 chars; reject before decoding an oversized blob.
    if len(payload) > (max_bytes * 4) // 3 + 4:
        raise OversizedInputError(f"pdf exceeds {max_bytes} bytes")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidPDFError("pdf_b64 is not valid base64") from exc
    return check_pdf_bytes(raw, max_bytes)


def check_pdf_bytes(raw: bytes, max_bytes: int) -> bytes:
    """Validate raw PDF bytes (size and magic number).

    Raises:
        InvalidPDFError: If the bytes do not start with ``%PDF``.
        OversizedInputError: If ``raw`` exceeds ``max_bytes``.
        EmptyInputError: If ``raw`` is empty.
    """
    if not raw:
        raise EmptyInputError("pdf payload is empty")
    if len(raw) > max_bytes:
        raise OversizedInputError(f"pdf exceeds {max_bytes} bytes")
    if not raw.lstrip()[:4].startswith(PDF_MAGIC):
        raise InvalidPDFError("payload is not a PDF (missing %PDF header)")
    return raw


def parse_pdf_bytes(pdf_bytes: bytes) -> ParsedDocument:
    """Extract the text layer of every page with pdfplumber.

    Args:
        pdf_bytes: A valid PDF.

    Returns:
        Parsed document with one :class:`PageText` per page. Pages without a text
        layer come back empty, which the router later treats as scanned.

    Raises:
        InvalidPDFError: If pdfplumber cannot open the file.
    """
    import pdfplumber  # lazy: keeps import time low for text-only requests

    pages: list[PageText] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                has_tables = bool(page.find_tables())
                pages.append(PageText(number=i, text=text, confidence=1.0, has_tables=has_tables))
    except Exception as exc:  # pdfminer raises many unrelated exception types
        logger.warning("pdfplumber failed to open document: {!r}", exc)
        raise InvalidPDFError(f"cannot open PDF: {exc.__class__.__name__}") from exc
    return ParsedDocument(pages=pages, source="text")


def parse_pdf_b64(pdf_b64: str, max_bytes: int = 10 * 1024 * 1024) -> ParsedDocument:
    """Decode a base64 PDF and parse its text layer (see :func:`parse_pdf_bytes`)."""
    return parse_pdf_bytes(decode_pdf_b64(pdf_b64, max_bytes))


def parse_text(raw: str, max_chars: int = 200_000) -> ParsedDocument:
    """Wrap already-extracted text as a single-page document.

    Args:
        raw: Document text.
        max_chars: Maximum accepted length.

    Raises:
        EmptyInputError: If the text is blank.
        OversizedInputError: If the text exceeds ``max_chars``.
    """
    if not raw or not raw.strip():
        raise EmptyInputError("text is empty")
    if len(raw) > max_chars:
        raise OversizedInputError(f"text exceeds {max_chars} characters")
    return ParsedDocument(pages=[PageText(number=1, text=raw, confidence=1.0)], source="raw")
