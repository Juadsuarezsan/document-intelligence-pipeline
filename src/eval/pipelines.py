"""The three pipelines compared in the mandatory table.

``heuristic``   "Unstructured only": pdfplumber/OCR text + rule-based extraction,
                no model call. Runs anywhere.
``fallback``    This system: heuristic first, Claude Vision re-processing when the
                confidence scorer routes to ``vlm_fallback``. Needs an API key.
``vision``      Claude Vision direct on page images. Needs an API key.

Without ``ANTHROPIC_API_KEY`` the two Claude pipelines are reported as
``pendiente`` rather than measured with a stand-in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from src.config import Settings
from src.eval.dataset import EvalCase
from src.extractors.heuristic import HeuristicExtractor
from src.llm.client import ClaudeClient
from src.parsers.ocr_parser import render_pdf_pages
from src.pipeline.graph import DocumentPipeline, build_deps
from src.schemas.document import PipelineResult
from src.synth.pdf import render_document_pdf, render_scanned_pdf

PipelineName = Literal["heuristic", "fallback", "vision"]
InputMode = Literal["text", "pdf", "scanned"]

PIPELINE_LABELS: dict[PipelineName, str] = {
    "heuristic": "Unstructured only (heurístico: pdfplumber/OCR + reglas, sin LLM)",
    "fallback": "Unstructured + Claude Vision fallback (este)",
    "vision": "Claude Vision directo",
}


@dataclass
class CaseOutcome:
    """What a pipeline produced for one case."""

    case_id: str
    mode: InputMode
    predicted: dict[str, Any]
    pairs: list[tuple[str, str]]
    predicted_type: str | None
    routing: str
    confidence: float
    latency_ms: int
    cost_usd: float
    pipeline_used: str
    findings: list[str]


def case_pdf(case: EvalCase) -> bytes:
    """Render the case to a text-layer PDF (with its table when present)."""
    if case.table:
        return render_document_pdf(case.text, case.table.headers, case.table.rows, case.table.ruled)
    return render_document_pdf(case.text)


def build_pipeline(name: PipelineName, settings: Settings) -> DocumentPipeline:
    """Instantiate the pipeline for a configuration.

    Raises:
        RuntimeError: For Claude pipelines without an API key.
    """
    if name == "heuristic":
        cfg = settings.model_copy(
            update={"anthropic_api_key": None, "use_claude_vision_fallback": False}
        )
        return DocumentPipeline(build_deps(cfg, ClaudeClient(cfg)))
    if not settings.llm_enabled:
        raise RuntimeError(f"pipeline {name!r} requires ANTHROPIC_API_KEY")
    cfg = settings.model_copy(update={"use_claude_vision_fallback": name == "fallback"})
    return DocumentPipeline(build_deps(cfg))


async def run_case(
    pipeline: DocumentPipeline,
    name: PipelineName,
    case: EvalCase,
    mode: InputMode,
    heuristic: HeuristicExtractor,
    deps_settings: Settings,
) -> CaseOutcome:
    """Run one case through a pipeline in the given input mode."""
    start = time.perf_counter()
    result: PipelineResult
    if name == "vision":
        result = await _run_vision_direct(pipeline, case, deps_settings)
    elif mode == "text":
        result = await pipeline.run(text=case.text)
    elif mode == "pdf":
        result = await pipeline.run(pdf_bytes=case_pdf(case))
    else:
        result = await pipeline.run(pdf_bytes=render_scanned_pdf(case_pdf(case)))
    latency_ms = int((time.perf_counter() - start) * 1000)
    predicted = {f.name: f.value for f in result.fields if f.value not in (None, "")}
    text_for_pairs = case.text if mode == "text" else _text_from_result(result, case)
    pairs = heuristic.extract_pairs(text_for_pairs) if case.source == "funsd" else []
    return CaseOutcome(
        case_id=case.id,
        mode=mode,
        predicted=predicted,
        pairs=pairs,
        predicted_type=result.document_type if result.classifier_method != "requested" else None,
        routing=result.routing,
        confidence=result.overall_confidence,
        latency_ms=latency_ms,
        cost_usd=result.usage.cost_usd,
        pipeline_used=result.pipeline_used,
        findings=[f.code for f in result.findings],
    )


def _text_from_result(result: PipelineResult, case: EvalCase) -> str:
    # Pair extraction needs the text the pipeline actually saw; the node log does
    # not carry it, so for PDF modes we re-parse deterministically.
    from src.parsers.text_parser import parse_pdf_bytes

    if result.pipeline_used == "text":
        return parse_pdf_bytes(case_pdf(case)).text
    return case.text


async def _run_vision_direct(
    pipeline: DocumentPipeline, case: EvalCase, settings: Settings
) -> PipelineResult:
    """Claude Vision direct: classify + extract from page images only."""
    from src.classifier.doc_type import DocTypeClassifier
    from src.extractors.llm_extractor import VLMExtractor
    from src.observability.tracing import RequestTrace
    from src.schemas.document import PipelineResult as _PR
    from src.scoring.confidence import document_confidence, routing_decision
    from src.validators.cross_field import validate_fields

    client = ClaudeClient(settings)
    trace = RequestTrace()
    images = render_pdf_pages(case_pdf(case))
    classification = await DocTypeClassifier(client).classify_image(images[0])
    trace.add_usage(classification.usage)
    extraction = await VLMExtractor(client).extract(images, classification.doc_type)
    trace.add_usage(extraction.usage)
    findings = validate_fields(extraction.fields, classification.doc_type)
    confidence = document_confidence(extraction.fields, findings, classification.doc_type)
    routing = routing_decision(
        confidence,
        auto_thr=settings.confidence_auto_approve,
        review_thr=settings.confidence_human_review,
    )
    _ = pipeline  # the graph is not used for the direct variant
    return _PR(
        trace_id=trace.trace_id,
        document_type=classification.doc_type,
        classifier_method="claude_vision",
        fields=extraction.fields,
        findings=findings,
        overall_confidence=confidence,
        routing=routing,
        pipeline_used="vlm",
        extractor_method="claude_vision",
        n_pages=len(images),
        latency_ms=trace.latency_ms,
        usage=trace.usage,
    )
