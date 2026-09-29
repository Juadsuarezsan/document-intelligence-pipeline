"""Table extraction.

* ``pdfplumber`` (default): ruled tables via its line/text strategies, no
  native dependencies.
* ``camelot`` stream mode (optional ``heavy`` extra): whitespace-aligned tables
  without ruling lines; needs OpenCV but not Ghostscript.
"""

from __future__ import annotations

import io
import tempfile
from typing import Literal

from loguru import logger

from src.schemas.document import ExtractedTable

TableBackend = Literal["pdfplumber", "camelot", "auto"]

_PLUMBER_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "intersection_tolerance": 5,
}
_PLUMBER_TEXT_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 3,
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
}


def _clean(cell: object) -> str:
    return " ".join(str(cell).split()) if cell is not None else ""


def _to_table(
    page: int, grid: list[list[object]], backend: Literal["pdfplumber", "camelot"], conf: float
) -> ExtractedTable | None:
    rows = [[_clean(c) for c in row] for row in grid if row and any(_clean(c) for c in row)]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return ExtractedTable(
        page=page, headers=rows[0], rows=rows[1:], confidence=conf, backend=backend
    )


def extract_tables_pdfplumber(pdf_bytes: bytes) -> list[ExtractedTable]:
    """Extract tables with pdfplumber, trying ruled lines first then text alignment.

    Args:
        pdf_bytes: Source PDF.

    Returns:
        Tables in page order. Confidence is 0.9 for ruled tables, 0.7 for
        text-aligned ones.
    """
    import pdfplumber  # lazy

    tables: list[ExtractedTable] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            found = page.extract_tables(table_settings=_PLUMBER_SETTINGS)
            conf = 0.9
            if not found:
                found = page.extract_tables(table_settings=_PLUMBER_TEXT_SETTINGS)
                conf = 0.7
            for grid in found:
                table = _to_table(number, [list(row) for row in grid], "pdfplumber", conf)
                if table:
                    tables.append(table)
    logger.debug("pdfplumber tables={} ", len(tables))
    return tables


def extract_tables_camelot(pdf_bytes: bytes) -> list[ExtractedTable]:
    """Extract tables with Camelot in ``stream`` mode (optional dependency).

    Raises:
        ImportError: If ``camelot`` is not installed.
    """
    import camelot  # lazy optional import

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        result = camelot.read_pdf(tmp.name, pages="all", flavor="stream")
    tables: list[ExtractedTable] = []
    for t in result:
        grid = t.df.values.tolist()
        conf = float(getattr(t, "accuracy", 70.0)) / 100.0
        table = _to_table(int(t.page), grid, "camelot", max(0.0, min(1.0, conf)))
        if table:
            tables.append(table)
    return tables


def extract_tables(pdf_bytes: bytes, backend: TableBackend = "auto") -> list[ExtractedTable]:
    """Extract tables with the requested backend.

    ``auto`` uses pdfplumber and only tries Camelot when pdfplumber finds
    nothing and Camelot is importable.
    """
    if backend == "camelot":
        return extract_tables_camelot(pdf_bytes)
    tables = extract_tables_pdfplumber(pdf_bytes)
    if tables or backend == "pdfplumber":
        return tables
    try:
        return extract_tables_camelot(pdf_bytes)
    except ImportError:
        logger.debug("camelot not installed; returning pdfplumber result")
        return tables
