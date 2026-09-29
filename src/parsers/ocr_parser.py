"""OCR path for scanned documents: rasterise pages with pdfium, read them with tesseract.

``pytesseract`` is imported lazily so the module imports (and the rest of the
pipeline runs) even when the tesseract binary is not installed.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass, field

from loguru import logger
from PIL import Image

from src.parsers.errors import InvalidPDFError
from src.schemas.document import PageText, ParsedDocument


class OCRUnavailableError(RuntimeError):
    """Raised when the tesseract binary cannot be found."""


def tesseract_available(cmd: str = "tesseract") -> bool:
    """Return True if a tesseract executable is reachable.

    Args:
        cmd: Path or name of the binary.
    """
    return shutil.which(cmd) is not None


def render_pdf_pages(pdf_bytes: bytes, dpi: int = 150, max_pages: int = 20) -> list[Image.Image]:
    """Rasterise PDF pages to RGB images.

    Args:
        pdf_bytes: Source PDF.
        dpi: Render resolution (150 is enough for tesseract on printed forms).
        max_pages: Safety cap on the number of pages rendered.

    Returns:
        One PIL image per page.

    Raises:
        InvalidPDFError: If pdfium cannot open the document.
    """
    import pypdfium2 as pdfium  # lazy

    try:
        doc = pdfium.PdfDocument(pdf_bytes)
    except Exception as exc:
        raise InvalidPDFError(f"cannot rasterise PDF: {exc.__class__.__name__}") from exc
    images: list[Image.Image] = []
    scale = dpi / 72.0
    for index in range(min(len(doc), max_pages)):
        page = doc[index]
        bitmap = page.render(scale=scale)
        images.append(bitmap.to_pil().convert("RGB"))
    return images


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Encode a PIL image as PNG bytes (used for Claude Vision blocks)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@dataclass
class OCRWord:
    """A recognised token with its tesseract confidence (0-100)."""

    text: str
    conf: float
    line: int
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class OCRPage:
    """OCR result for one page."""

    text: str
    words: list[OCRWord] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        """Mean word confidence scaled to 0-1 (0.0 when nothing was recognised)."""
        confs = [w.conf for w in self.words if w.conf >= 0]
        return (sum(confs) / len(confs)) / 100.0 if confs else 0.0


def _assemble_lines(data: dict[str, list[object]]) -> OCRPage:
    """Turn ``image_to_data`` dict output into lines of text plus word confidences."""
    n = len(data.get("text", []))
    words: list[OCRWord] = []
    lines: dict[tuple[int, int, int], list[str]] = {}
    for i in range(n):
        token = str(data["text"][i]).strip()
        if not token:
            continue
        conf = float(str(data["conf"][i]))
        key = (
            int(str(data["block_num"][i])),
            int(str(data["par_num"][i])),
            int(str(data["line_num"][i])),
        )
        lines.setdefault(key, []).append(token)
        bbox = (
            int(str(data["left"][i])),
            int(str(data["top"][i])),
            int(str(data["width"][i])),
            int(str(data["height"][i])),
        )
        words.append(OCRWord(text=token, conf=conf, line=len(lines), bbox=bbox))
    text = "\n".join(" ".join(tokens) for _, tokens in sorted(lines.items()))
    return OCRPage(text=text, words=words)


def ocr_image(image: Image.Image, tesseract_cmd: str = "tesseract", lang: str = "eng") -> OCRPage:
    """Run tesseract on one image and return text with word confidences.

    Args:
        image: Page image.
        tesseract_cmd: Path to the tesseract binary.
        lang: Tesseract language pack.

    Raises:
        OCRUnavailableError: If the binary is missing.
    """
    if not tesseract_available(tesseract_cmd):
        raise OCRUnavailableError(f"tesseract binary not found at {tesseract_cmd!r}")
    import pytesseract  # lazy

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    data = pytesseract.image_to_data(
        image, lang=lang, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    page = _assemble_lines(data)
    logger.debug("ocr page words={} mean_conf={:.2f}", len(page.words), page.mean_confidence)
    return page


def ocr_images(images: list[Image.Image], tesseract_cmd: str = "tesseract") -> ParsedDocument:
    """OCR a list of page images into a :class:`ParsedDocument`."""
    pages = []
    for i, img in enumerate(images, start=1):
        result = ocr_image(img, tesseract_cmd=tesseract_cmd)
        pages.append(PageText(number=i, text=result.text, confidence=result.mean_confidence))
    return ParsedDocument(pages=pages, source="ocr")


def ocr_pdf_bytes(
    pdf_bytes: bytes, tesseract_cmd: str = "tesseract", dpi: int = 150
) -> ParsedDocument:
    """Rasterise and OCR a whole PDF."""
    return ocr_images(render_pdf_pages(pdf_bytes, dpi=dpi), tesseract_cmd=tesseract_cmd)


def ocr_image_bytes(image_bytes: bytes, tesseract_cmd: str = "tesseract") -> ParsedDocument:
    """OCR a single encoded image (PNG/JPEG)."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, ValueError) as exc:
        raise InvalidPDFError("payload is not a decodable image") from exc
    return ocr_images([image], tesseract_cmd=tesseract_cmd)
