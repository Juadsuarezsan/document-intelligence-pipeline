"""Render synthetic documents to PDF with reportlab (text layer, tables, scans)."""

from __future__ import annotations

import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.parsers.ocr_parser import render_pdf_pages

_LINE_HEIGHT = 14
_MARGIN = 0.9 * inch


def render_text_pdf(text: str, font: str = "Helvetica", size: int = 10) -> bytes:
    """Render plain text line by line into a text-layer PDF.

    Args:
        text: Document text (newline separated).
        font: Base font name.
        size: Font size in points.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    y = height - _MARGIN
    c.setFont(font, size)
    for line in text.splitlines():
        if y < _MARGIN:
            c.showPage()
            c.setFont(font, size)
            y = height - _MARGIN
        c.drawString(_MARGIN, y, line[:120])
        y -= _LINE_HEIGHT
    c.showPage()
    c.save()
    return buf.getvalue()


def render_table_pdf(
    headers: list[str], rows: list[list[str]], ruled: bool = True, intro: str = ""
) -> bytes:
    """Render a single table (optionally with ruling lines) into a PDF."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=_MARGIN, rightMargin=_MARGIN)
    styles = getSampleStyleSheet()
    story: list[object] = []
    if intro:
        story.append(Paragraph(intro, styles["Normal"]))
        story.append(Spacer(1, 12))
    table = Table([headers] + rows, hAlign="LEFT")
    style: list[tuple[Any, ...]] = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if ruled:
        style += [("GRID", (0, 0), (-1, -1), 0.6, colors.black)]
    else:
        style += [("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black)]
    table.setStyle(TableStyle(style))
    story.append(table)
    doc.build(story)
    return buf.getvalue()


def render_document_pdf(
    text: str,
    headers: list[str] | None = None,
    rows: list[list[str]] | None = None,
    ruled: bool = True,
) -> bytes:
    """Render a document as text lines followed by an optional table on the same page."""
    if not headers or not rows:
        return render_text_pdf(text)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=_MARGIN, rightMargin=_MARGIN)
    styles = getSampleStyleSheet()
    body = styles["Normal"]
    body.fontName = "Helvetica"
    body.fontSize = 10
    body.leading = 13
    story: list[object] = []
    for line in text.splitlines():
        story.append(Paragraph(line.replace("&", "&amp;") or "&nbsp;", body))
    story.append(Spacer(1, 10))
    table = Table([headers] + rows, hAlign="LEFT")
    style: list[tuple[Any, ...]] = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]
    style += (
        [("GRID", (0, 0), (-1, -1), 0.6, colors.black)]
        if ruled
        else [("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black)]
    )
    table.setStyle(TableStyle(style))
    story.append(table)
    doc.build(story)
    return buf.getvalue()


def render_scanned_pdf(pdf_bytes: bytes, dpi: int = 150) -> bytes:
    """Turn a text PDF into an image-only PDF (simulates a scanner: no text layer)."""
    images = render_pdf_pages(pdf_bytes, dpi=dpi)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    width, height = LETTER
    for img in images:
        c.drawImage(ImageReader(img), 0, 0, width=width, height=height)
        c.showPage()
    c.save()
    return buf.getvalue()
