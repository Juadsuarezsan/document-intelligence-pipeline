import pytest

from src.config import Settings
from src.llm.client import ClaudeClient
from src.observability.tracing import RequestTrace
from src.parsers.errors import EmptyInputError
from src.parsers.ocr_parser import tesseract_available
from src.pipeline.graph import DocumentPipeline, build_deps
from src.synth.pdf import render_scanned_pdf, render_text_pdf
from tests.conftest import FakeAnthropic


def _heuristic_pipeline(settings: Settings) -> DocumentPipeline:
    return DocumentPipeline(build_deps(settings))


async def test_text_path_end_to_end(settings: Settings, invoice_text: str) -> None:
    result = await _heuristic_pipeline(settings).run(text=invoice_text)
    assert result.document_type == "invoice" and result.classifier_method == "heuristic"
    assert result.pipeline_used == "text" and result.extractor_method == "heuristic"
    names = {f.name for f in result.fields}
    assert {"invoice_number", "total", "subtotal", "tax"} <= names
    assert result.routing in {"auto_approve", "human_review", "vlm_fallback"}
    assert [n["node"] for n in result.node_log] == [
        "n_ingest",
        "n_classify",
        "n_tables",
        "n_extract",
        "n_validate",
        "n_score",
    ]
    assert result.usage.llm_calls == 0 and result.usage.cost_usd == 0.0
    assert len(result.trace_id) == 32


async def test_pdf_path_extracts_tables(settings: Settings, invoice_pdf: bytes) -> None:
    result = await _heuristic_pipeline(settings).run(pdf_bytes=invoice_pdf, document_type="invoice")
    assert result.classifier_method == "requested"
    assert result.pipeline_used == "text" and result.n_pages == 1
    assert len(result.tables) == 1 and result.tables[0].headers[0] == "Description"


async def test_empty_text_raises(settings: Settings) -> None:
    with pytest.raises(EmptyInputError):
        await _heuristic_pipeline(settings).run(text="   ")


async def test_pii_is_redacted_and_reported(settings: Settings) -> None:
    text = "APPLICATION FORM\nForm ID: F-1\nName: Ana ana@example.org\nDate: 2026-01-01\n"
    result = await _heuristic_pipeline(settings).run(text=text, document_type="form")
    submitter = next(f for f in result.fields if f.name == "submitter")
    assert submitter.redacted and "[REDACTED:email]" in str(submitter.value)
    assert any(f.code == "pii_redacted" for f in result.findings)


@pytest.mark.skipif(not tesseract_available("tesseract"), reason="tesseract binary not installed")
async def test_scanned_path_uses_ocr(invoice_text: str) -> None:
    settings = Settings(ANTHROPIC_API_KEY=None, TESSERACT_CMD="tesseract")
    scanned = render_scanned_pdf(render_text_pdf(invoice_text))
    result = await _heuristic_pipeline(settings).run(pdf_bytes=scanned)
    assert result.pipeline_used == "ocr"
    assert result.document_type == "invoice"
    assert any(f.name == "invoice_number" for f in result.fields)


async def test_force_vlm_path_with_mocked_vision(
    fake_anthropic: FakeAnthropic, invoice_pdf: bytes
) -> None:
    cfg = Settings(ANTHROPIC_API_KEY="test-key", USE_CLAUDE_VISION_FALLBACK=True)
    llm = ClaudeClient(cfg, client=fake_anthropic)  # type: ignore[arg-type]
    fake_anthropic.queue(
        '{"pages": [{"text": "Invoice No: INV-9\\nTotal Due: USD 10.00", "confidence": 0.9}]}',  # transcription
        '{"type": "invoice", "confidence": 0.95}',  # classification
        '{"fields": [{"name": "invoice_number", "value": "INV-9", "confidence": 0.99}, {"name": "total", "value": 10, "confidence": 0.99}]}',
        '{"fields": [{"name": "invoice_number", "value": "INV-9", "confidence": 0.99}, {"name": "total", "value": 10, "confidence": 0.99}, {"name": "vendor", "value": "Acme", "confidence": 0.9}]}',
    )
    pipeline = DocumentPipeline(build_deps(cfg, llm))
    result = await pipeline.run(pdf_bytes=invoice_pdf, force_vlm=True)
    assert result.pipeline_used == "vlm"
    assert result.classifier_method == "claude" and result.extractor_method == "claude_vision"
    assert result.vlm_reprocessed
    assert result.usage.llm_calls == 4 and result.usage.cost_usd > 0
    assert {f.name for f in result.fields} == {"invoice_number", "total", "vendor"}


async def test_low_confidence_triggers_vision_reprocess(
    fake_anthropic: FakeAnthropic, invoice_pdf: bytes
) -> None:
    # Thresholds so high that any first pass routes to vlm_fallback.
    cfg = Settings(
        ANTHROPIC_API_KEY="test-key", CONFIDENCE_AUTO_APPROVE=1.0, CONFIDENCE_HUMAN_REVIEW=0.999
    )
    llm = ClaudeClient(cfg, client=fake_anthropic)  # type: ignore[arg-type]
    fake_anthropic.queue(
        '{"type": "invoice", "confidence": 0.9}',  # classify (text path)
        '{"fields": [{"name": "invoice_number", "value": "INV-1", "confidence": 0.5}]}',  # extract
        '{"fields": [{"name": "invoice_number", "value": "INV-1", "confidence": 0.9}, {"name": "total", "value": 2499, "confidence": 0.9}]}',  # vision reprocess
    )
    result = await DocumentPipeline(build_deps(cfg, llm)).run(pdf_bytes=invoice_pdf)
    assert result.vlm_reprocessed and result.extractor_method == "claude_vision"
    assert "n_vlm_reprocess" in [n["node"] for n in result.node_log]
    assert (
        result.node_log.count(
            {
                "node": "n_validate",
                "status": "ok",
                "ms": result.node_log[4]["ms"],
                **{
                    k: v for k, v in result.node_log[4].items() if k not in ("node", "status", "ms")
                },
            }
        )
        >= 1
    )
    assert result.usage.llm_calls == 3


async def test_vision_reprocess_survives_bad_output(
    fake_anthropic: FakeAnthropic, invoice_pdf: bytes
) -> None:
    cfg = Settings(
        ANTHROPIC_API_KEY="test-key", CONFIDENCE_AUTO_APPROVE=1.0, CONFIDENCE_HUMAN_REVIEW=0.999
    )
    llm = ClaudeClient(cfg, client=fake_anthropic)  # type: ignore[arg-type]
    fake_anthropic.queue('{"type": "invoice"}', '{"fields": []}', "garbage")
    result = await DocumentPipeline(build_deps(cfg, llm)).run(pdf_bytes=invoice_pdf)
    assert result.vlm_reprocessed and result.extractor_method == "claude"
    assert result.routing == "vlm_fallback"


async def test_trace_can_be_supplied(settings: Settings, invoice_text: str) -> None:
    trace = RequestTrace(trace_id="abc123")
    result = await _heuristic_pipeline(settings).run(text=invoice_text, trace=trace)
    assert result.trace_id == "abc123"
    assert result.latency_ms >= 0 and trace.summary()["trace_id"] == "abc123"
