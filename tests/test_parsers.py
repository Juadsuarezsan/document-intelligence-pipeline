import base64

import pytest

from src.parsers.errors import EmptyInputError, InvalidPDFError, OversizedInputError
from src.parsers.layout import classify_line, key_value_pairs, partition, split_multi_kv
from src.parsers.router import choose_input_path
from src.parsers.text_parser import check_pdf_bytes, decode_pdf_b64, parse_pdf_bytes, parse_text
from src.schemas.document import PageText, ParsedDocument


def test_decode_pdf_b64_roundtrip(invoice_pdf: bytes) -> None:
    b64 = base64.b64encode(invoice_pdf).decode()
    assert decode_pdf_b64(b64, max_bytes=10_000_000) == invoice_pdf
    assert decode_pdf_b64("data:application/pdf;base64," + b64, max_bytes=10_000_000) == invoice_pdf


def test_decode_pdf_b64_rejects_empty_invalid_and_oversized(invoice_pdf: bytes) -> None:
    with pytest.raises(EmptyInputError):
        decode_pdf_b64("   ", max_bytes=100)
    with pytest.raises(InvalidPDFError):
        decode_pdf_b64("not base64 at all!!", max_bytes=100)
    with pytest.raises(InvalidPDFError):
        decode_pdf_b64(base64.b64encode(b"hello world").decode(), max_bytes=100)
    with pytest.raises(OversizedInputError):
        decode_pdf_b64(base64.b64encode(invoice_pdf).decode(), max_bytes=len(invoice_pdf) - 1)


def test_check_pdf_bytes_errors() -> None:
    with pytest.raises(EmptyInputError):
        check_pdf_bytes(b"", 10)
    with pytest.raises(OversizedInputError):
        check_pdf_bytes(b"%PDF-1.4 " + b"x" * 20, 10)
    with pytest.raises(InvalidPDFError):
        check_pdf_bytes(b"GIF89a", 100)


def test_parse_pdf_bytes_extracts_text_and_tables(invoice_pdf: bytes) -> None:
    doc = parse_pdf_bytes(invoice_pdf)
    assert doc.n_pages == 1
    assert "INV-2026-0042" in doc.text
    assert doc.has_tables
    assert not doc.is_scanned(min_chars_per_page=20)


def test_parse_pdf_bytes_rejects_garbage() -> None:
    with pytest.raises(InvalidPDFError):
        parse_pdf_bytes(b"%PDF-1.4 this is not really a pdf")


def test_parse_text_limits() -> None:
    doc = parse_text("hello")
    assert doc.text == "hello" and doc.source == "raw"
    with pytest.raises(EmptyInputError):
        parse_text("  \n ")
    with pytest.raises(OversizedInputError):
        parse_text("x" * 11, max_chars=10)


def test_layout_classification() -> None:
    assert classify_line("Invoice No: INV-1").category == "KeyValue"
    assert classify_line("3. Governing Law: Delaware").key == "Governing Law"
    assert classify_line("Tax (19%): 399.00").key == "Tax (19%)"
    assert classify_line("ACME CONSULTING S.A.S.").category == "Title"
    assert classify_line("- first item").category == "ListItem"
    assert classify_line("Revenue    1,000    1,200").category == "Table"
    assert classify_line("This is a normal sentence about things.").category == "NarrativeText"
    assert classify_line("   ").category == "Other"


def test_split_multi_kv_handles_funsd_reading_order() -> None:
    elements = split_multi_kv("To:  Mr. Smith  Date:  9/3/92")
    assert [(e.key, e.value) for e in elements] == [("To", "Mr. Smith"), ("Date", "9/3/92")]
    assert split_multi_kv("just   some   words") == []
    pairs = key_value_pairs(partition("Name:  Alice  Dept:  Finance\nCity: Bogotá"))
    assert pairs == [("Name", "Alice"), ("Dept", "Finance"), ("City", "Bogotá")]


def test_partition_falls_back_when_unstructured_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("unstructured"):
            raise ImportError("no unstructured")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    elements = partition("Invoice: 1", use_unstructured=True)
    assert elements[0].category == "KeyValue"


def _doc(chars: int) -> ParsedDocument:
    return ParsedDocument(pages=[PageText(number=1, text="x" * chars)], source="text")


def test_router_decisions() -> None:
    common = {"min_chars_per_page": 20}
    assert (
        choose_input_path(None, force_vlm=False, vlm_enabled=False, ocr_available=True, **common)
        == "text"
    )
    assert (
        choose_input_path(
            _doc(500), force_vlm=False, vlm_enabled=True, ocr_available=True, **common
        )
        == "text"
    )
    assert (
        choose_input_path(_doc(0), force_vlm=False, vlm_enabled=False, ocr_available=True, **common)
        == "ocr"
    )
    assert (
        choose_input_path(_doc(0), force_vlm=False, vlm_enabled=True, ocr_available=False, **common)
        == "vlm"
    )
    assert (
        choose_input_path(
            _doc(0), force_vlm=False, vlm_enabled=False, ocr_available=False, **common
        )
        == "text"
    )
    assert (
        choose_input_path(_doc(500), force_vlm=True, vlm_enabled=True, ocr_available=True, **common)
        == "vlm"
    )
    assert (
        choose_input_path(
            _doc(500), force_vlm=True, vlm_enabled=False, ocr_available=True, **common
        )
        == "text"
    )
