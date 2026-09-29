import pytest
from PIL import Image

from src.parsers.ocr_parser import (
    OCRUnavailableError,
    _assemble_lines,
    image_to_png_bytes,
    ocr_image,
    ocr_image_bytes,
    ocr_images,
    ocr_pdf_bytes,
    render_pdf_pages,
    tesseract_available,
)
from src.synth.pdf import render_scanned_pdf, render_text_pdf

FAKE_DATA = {
    "text": ["Invoice", "No:", "INV-1", "", "Total:", "42.00"],
    "conf": ["96", "95", "93", "-1", "90", "88"],
    "block_num": [1, 1, 1, 1, 1, 1],
    "par_num": [1, 1, 1, 1, 1, 1],
    "line_num": [1, 1, 1, 1, 2, 2],
    "left": [0, 10, 20, 0, 0, 10],
    "top": [0, 0, 0, 0, 20, 20],
    "width": [5, 5, 5, 0, 5, 5],
    "height": [5, 5, 5, 0, 5, 5],
}


def test_assemble_lines_groups_words_and_confidence() -> None:
    page = _assemble_lines(FAKE_DATA)  # type: ignore[arg-type]
    assert page.text == "Invoice No: INV-1\nTotal: 42.00"
    assert len(page.words) == 5
    assert 0.9 < page.mean_confidence < 0.93


def test_ocr_image_with_mocked_tesseract(mocker: pytest.MonkeyPatch) -> None:
    import pytesseract

    mocker.patch("src.parsers.ocr_parser.tesseract_available", return_value=True)  # type: ignore[attr-defined]
    mocker.patch.object(pytesseract, "image_to_data", return_value=FAKE_DATA)  # type: ignore[attr-defined]
    img = Image.new("RGB", (100, 40), "white")
    page = ocr_image(img, tesseract_cmd="/nonexistent/tesseract")
    assert "INV-1" in page.text
    doc = ocr_images([img, img], tesseract_cmd="/nonexistent/tesseract")
    assert doc.n_pages == 2 and doc.source == "ocr"
    assert 0.9 < doc.mean_confidence < 0.93


def test_ocr_image_raises_when_binary_missing() -> None:
    assert not tesseract_available("/definitely/not/here")
    with pytest.raises(OCRUnavailableError):
        ocr_image(Image.new("RGB", (10, 10)), tesseract_cmd="/definitely/not/here")


def test_render_pdf_pages_and_png_bytes() -> None:
    pdf = render_text_pdf("Hello OCR\nSecond line")
    images = render_pdf_pages(pdf, dpi=72)
    assert len(images) == 1
    assert images[0].size[0] > 500
    assert image_to_png_bytes(images[0]).startswith(b"\x89PNG")


def test_ocr_image_bytes_rejects_non_image() -> None:
    from src.parsers.errors import InvalidPDFError

    with pytest.raises(InvalidPDFError):
        ocr_image_bytes(b"not an image", tesseract_cmd="tesseract")


@pytest.mark.requires_tesseract
@pytest.mark.skipif(not tesseract_available("tesseract"), reason="tesseract binary not installed")
def test_real_tesseract_reads_scanned_pdf(invoice_text: str) -> None:
    scanned = render_scanned_pdf(render_text_pdf(invoice_text))
    from src.parsers.text_parser import parse_pdf_bytes

    assert parse_pdf_bytes(scanned).is_scanned(min_chars_per_page=20)
    doc = ocr_pdf_bytes(scanned, tesseract_cmd="tesseract")
    assert "INV-2026-0042" in doc.text
    assert doc.mean_confidence > 0.7
