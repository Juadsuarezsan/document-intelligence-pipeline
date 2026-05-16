"""Text-PDF parser using pdfplumber. Returns plain text + table candidates."""
from __future__ import annotations

import base64
from dataclasses import dataclass


@dataclass
class ParsedText:
    text: str
    n_pages: int
    has_tables: bool


def parse_pdf_b64(pdf_b64: str) -> ParsedText:
    import io
    import pdfplumber  # lazy import
    pdf_bytes = base64.b64decode(pdf_b64)
    pages: list[str] = []
    has_tables = False
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
            if page.find_tables():
                has_tables = True
    return ParsedText(text="\n\n".join(pages), n_pages=len(pages), has_tables=has_tables)


def parse_text(raw: str) -> ParsedText:
    return ParsedText(text=raw, n_pages=1, has_tables=False)
