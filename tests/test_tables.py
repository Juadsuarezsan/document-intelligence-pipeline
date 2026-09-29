import sys
import types

import pytest

from src.eval.metrics import cell_prf
from src.tables.extractor import extract_tables, extract_tables_camelot, extract_tables_pdfplumber


def test_pdfplumber_extracts_ruled_table(table_pdf: bytes) -> None:
    tables = extract_tables_pdfplumber(table_pdf)
    assert len(tables) == 1
    t = tables[0]
    assert t.headers == ["Line", "FY2024", "FY2025"]
    assert t.rows == [["Revenue", "1,000", "1,200"], ["Opex", "600", "650"]]
    assert t.backend == "pdfplumber" and t.confidence == 0.9
    assert t.n_cells == 9
    assert (
        cell_prf(
            t.headers,
            t.rows,
            ["Line", "FY2024", "FY2025"],
            [["Revenue", "1,000", "1,200"], ["Opex", "600", "650"]],
        ).f1
        == 1.0
    )


def test_cell_prf_counts_mismatches() -> None:
    prf = cell_prf(["a", "b"], [["1", "x"], ["extra", "row"]], ["a", "b"], [["1", "2"]])
    assert prf.tp == 3 and prf.fn == 1 and prf.fp == 3
    assert 0 < prf.f1 < 1
    empty = cell_prf([], [], ["a"], [["1"]])
    assert empty.tp == 0 and empty.fn == 2 and empty.f1 == 0.0


def test_extract_tables_auto_falls_back_without_camelot(
    table_pdf: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "camelot", None)  # import raises ImportError
    from src.synth.pdf import render_text_pdf

    assert extract_tables(render_text_pdf("no tables here"), backend="auto") == []
    assert len(extract_tables(table_pdf, backend="pdfplumber")) == 1


def test_camelot_backend_with_fake_module(
    table_pdf: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeDF:
        values = types.SimpleNamespace(tolist=lambda: [["Line", "FY2024"], ["Revenue", "1,000"]])

    class FakeTable:
        df = FakeDF()
        page = 1
        accuracy = 85.0

    fake = types.ModuleType("camelot")
    fake.read_pdf = lambda *a, **k: [FakeTable()]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "camelot", fake)
    tables = extract_tables_camelot(table_pdf)
    assert tables[0].backend == "camelot" and tables[0].confidence == pytest.approx(0.85)
    assert tables[0].headers == ["Line", "FY2024"]
    assert extract_tables(table_pdf, backend="camelot")[0].backend == "camelot"
