"""The document pipeline as a LangGraph ``StateGraph``.

Nodes (names are prefixed ``n_`` so they can never collide with state keys):

    n_ingest -> n_classify -> n_tables -> n_extract -> n_validate -> n_score
                                                              |
                        (routing == vlm_fallback and Vision available)
                                                              v
                                          n_vlm_reprocess -> n_validate -> n_score -> END

Every node logs its input/output summary under the request ``trace_id`` and
records its duration in ``node_log``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
from loguru import logger
from PIL import Image

from src.classifier.doc_type import DocTypeClassifier
from src.config import Settings
from src.extractors.llm_extractor import LLMExtractor, VLMExtractor
from src.llm.client import ClaudeClient, LLMOutputError
from src.observability.tracing import RequestTrace
from src.parsers.ocr_parser import ocr_images, render_pdf_pages, tesseract_available
from src.parsers.router import choose_input_path
from src.parsers.text_parser import parse_pdf_bytes, parse_text
from src.parsers.vlm_parser import VLMParser
from src.schemas.document import (
    DocumentType,
    ExtractedField,
    ExtractedTable,
    Finding,
    InputPath,
    ParsedDocument,
    PipelineResult,
    Routing,
    UsageStats,
)
from src.scoring.confidence import document_confidence, penalize_fields, routing_decision
from src.tables.extractor import extract_tables
from src.validators.cross_field import validate_fields
from src.validators.pii import redact_fields


class PipelineState(TypedDict, total=False):
    """Mutable state threaded through the graph."""

    pdf_bytes: bytes | None
    text: str | None
    requested_type: DocumentType | None
    force_vlm: bool
    parsed: ParsedDocument
    images: list[Image.Image]
    pipeline_used: InputPath
    doc_type: DocumentType
    classifier_method: Literal["requested", "heuristic", "claude", "claude_vision"]
    fields: list[ExtractedField]
    extractor_method: Literal["heuristic", "claude", "claude_vision"]
    tables: list[ExtractedTable]
    findings: list[Finding]
    confidence: float
    routing: Routing
    vlm_reprocessed: bool


@dataclass
class PipelineDeps:
    """Everything the nodes need, built once at startup."""

    settings: Settings
    classifier: DocTypeClassifier
    extractor: LLMExtractor
    vlm_parser: VLMParser | None = None
    vlm_extractor: VLMExtractor | None = None

    @property
    def vlm_enabled(self) -> bool:
        """Vision fallback is configured and has credentials."""
        return (
            self.settings.use_claude_vision_fallback
            and self.vlm_parser is not None
            and self.vlm_parser.enabled
        )


def build_deps(settings: Settings, llm: ClaudeClient | None = None) -> PipelineDeps:
    """Construct the dependency bundle from settings (Claude parts only with a key)."""
    client = llm if llm is not None else ClaudeClient(settings)
    vlm_parser = VLMParser(client) if client.enabled else None
    vlm_extractor = VLMExtractor(client) if client.enabled else None
    return PipelineDeps(
        settings=settings,
        classifier=DocTypeClassifier(client if client.enabled else None),
        extractor=LLMExtractor(client if client.enabled else None),
        vlm_parser=vlm_parser,
        vlm_extractor=vlm_extractor,
    )


class DocumentPipeline:
    """Compiles the graph once and runs it per request with a fresh trace."""

    def __init__(self, deps: PipelineDeps) -> None:
        self._deps = deps
        self._graph = self._build()

    # ----------------------------------------------------------------- nodes
    async def _n_ingest(self, state: PipelineState, trace: RequestTrace) -> dict[str, Any]:
        s = self._deps.settings
        with trace.span("n_ingest", has_pdf=state.get("pdf_bytes") is not None) as out:
            pdf = state.get("pdf_bytes")
            images: list[Image.Image] = []
            if pdf is not None:
                parsed = parse_pdf_bytes(pdf)
                path = choose_input_path(
                    parsed,
                    force_vlm=state.get("force_vlm", False),
                    vlm_enabled=self._deps.vlm_enabled,
                    ocr_available=tesseract_available(s.tesseract_cmd),
                    min_chars_per_page=s.ocr_min_chars_per_page,
                )
                if path in ("ocr", "vlm") or self._deps.vlm_enabled:
                    images = render_pdf_pages(pdf)
                if path == "ocr":
                    parsed = ocr_images(images, tesseract_cmd=s.tesseract_cmd)
                elif path == "vlm" and self._deps.vlm_parser is not None:
                    parsed, usage = await self._deps.vlm_parser.transcribe(images)
                    trace.add_usage(usage)
            else:
                parsed = parse_text(state.get("text") or "", max_chars=s.max_text_chars)
                path = "text"
            out.update(path=path, pages=parsed.n_pages, chars=len(parsed.text))
            return {"parsed": parsed, "images": images, "pipeline_used": path}

    async def _n_classify(self, state: PipelineState, trace: RequestTrace) -> dict[str, Any]:
        with trace.span("n_classify", requested=state.get("requested_type")) as out:
            requested = state.get("requested_type")
            if requested is not None:
                out.update(doc_type=requested, method="requested")
                return {"doc_type": requested, "classifier_method": "requested"}
            parsed = state["parsed"]
            if parsed.text.strip():
                result = await self._deps.classifier.classify(parsed.text)
            else:
                result = await self._deps.classifier.classify_image(state["images"][0])
            trace.add_usage(result.usage)
            out.update(doc_type=result.doc_type, method=result.method, conf=result.confidence)
            return {"doc_type": result.doc_type, "classifier_method": result.method}

    async def _n_tables(self, state: PipelineState, trace: RequestTrace) -> dict[str, Any]:
        with trace.span("n_tables") as out:
            pdf = state.get("pdf_bytes")
            tables = extract_tables(pdf) if pdf and state["parsed"].has_tables else []
            out.update(tables=len(tables))
            return {"tables": tables}

    async def _n_extract(self, state: PipelineState, trace: RequestTrace) -> dict[str, Any]:
        with trace.span("n_extract", doc_type=state["doc_type"]) as out:
            result = await self._deps.extractor.extract(state["parsed"].text, state["doc_type"])
            trace.add_usage(result.usage)
            method: Literal["heuristic", "claude"] = (
                "claude" if result.method == "claude" else "heuristic"
            )
            out.update(fields=len(result.fields), method=result.method)
            return {"fields": result.fields, "extractor_method": method}

    async def _n_validate(self, state: PipelineState, trace: RequestTrace) -> dict[str, Any]:
        with trace.span("n_validate", fields=len(state["fields"])) as out:
            findings = validate_fields(state["fields"], state["doc_type"])
            if state["parsed"].source == "ocr" and state["parsed"].mean_confidence < 0.6:
                findings.append(
                    Finding(
                        severity="warn",
                        code="low_ocr_confidence",
                        message="OCR mean word confidence below 0.6",
                        details={"mean_confidence": round(state["parsed"].mean_confidence, 3)},
                    )
                )
            fields = penalize_fields(state["fields"], findings)
            if self._deps.settings.redact_pii:
                fields, report = redact_fields(fields)
                if report.total:
                    findings.append(
                        Finding(
                            severity="info",
                            code="pii_redacted",
                            message=f"{report.total} PII span(s) masked",
                            details={k: v for k, v in report.counts.items()},
                        )
                    )
            out.update(
                findings=len(findings), errors=sum(1 for f in findings if f.severity == "error")
            )
            return {"findings": findings, "fields": fields}

    async def _n_score(self, state: PipelineState, trace: RequestTrace) -> dict[str, Any]:
        s = self._deps.settings
        with trace.span("n_score") as out:
            confidence = document_confidence(state["fields"], state["findings"], state["doc_type"])
            routing = routing_decision(
                confidence, auto_thr=s.confidence_auto_approve, review_thr=s.confidence_human_review
            )
            out.update(confidence=confidence, routing=routing)
            return {"confidence": confidence, "routing": routing}

    async def _n_vlm_reprocess(self, state: PipelineState, trace: RequestTrace) -> dict[str, Any]:
        with trace.span("n_vlm_reprocess", pages=len(state.get("images", []))) as out:
            extractor = self._deps.vlm_extractor
            images = state.get("images", [])
            if extractor is None or not images:
                out.update(skipped="no_vision_or_images")
                return {"vlm_reprocessed": True}
            try:
                result = await extractor.extract(images, state["doc_type"])
            except LLMOutputError as exc:
                logger.warning("vision reprocess produced unusable output: {}", exc)
                out.update(skipped="bad_output")
                return {"vlm_reprocessed": True}
            trace.add_usage(result.usage)
            out.update(fields=len(result.fields))
            return {
                "fields": result.fields,
                "extractor_method": "claude_vision",
                "vlm_reprocessed": True,
            }

    # ---------------------------------------------------------------- wiring
    def _should_reprocess(self, state: PipelineState) -> str:
        if (
            state.get("routing") == "vlm_fallback"
            and not state.get("vlm_reprocessed", False)
            and self._deps.vlm_enabled
            and state.get("images")
        ):
            return "n_vlm_reprocess"
        return END

    def _build(self) -> Any:
        graph: StateGraph = StateGraph(PipelineState)
        # The trace is injected per run through the config, see `run`.
        for name in (
            "n_ingest",
            "n_classify",
            "n_tables",
            "n_extract",
            "n_validate",
            "n_score",
            "n_vlm_reprocess",
        ):
            graph.add_node(name, self._wrap(name))
        graph.set_entry_point("n_ingest")
        graph.add_edge("n_ingest", "n_classify")
        graph.add_edge("n_classify", "n_tables")
        graph.add_edge("n_tables", "n_extract")
        graph.add_edge("n_extract", "n_validate")
        graph.add_edge("n_validate", "n_score")
        graph.add_conditional_edges("n_score", self._should_reprocess, ["n_vlm_reprocess", END])
        graph.add_edge("n_vlm_reprocess", "n_validate")
        return graph.compile()

    def _wrap(self, name: str) -> Any:
        method = getattr(self, f"_{name}")

        async def node(state: PipelineState, config: dict[str, Any]) -> dict[str, Any]:
            trace: RequestTrace = config["configurable"]["trace"]
            return await method(state, trace)  # type: ignore[no-any-return]

        node.__name__ = name
        return node

    # ------------------------------------------------------------------- run
    async def run(
        self,
        *,
        text: str | None = None,
        pdf_bytes: bytes | None = None,
        document_type: DocumentType | None = None,
        force_vlm: bool = False,
        trace: RequestTrace | None = None,
    ) -> PipelineResult:
        """Process one document end to end.

        Args:
            text: Pre-extracted text (used when ``pdf_bytes`` is ``None``).
            pdf_bytes: Raw PDF (validated by the caller).
            document_type: Skip classification and use this type.
            force_vlm: Route straight to Claude Vision when available.
            trace: Optional trace to reuse (a new one is created otherwise).

        Returns:
            The pipeline result with fields, tables, findings, routing and usage.
        """
        trace = trace or RequestTrace()
        initial: PipelineState = {
            "pdf_bytes": pdf_bytes,
            "text": text,
            "requested_type": document_type,
            "force_vlm": force_vlm,
            "vlm_reprocessed": False,
        }
        final: PipelineState = await self._graph.ainvoke(
            initial, config={"configurable": {"trace": trace}, "recursion_limit": 25}
        )
        usage: UsageStats = trace.usage
        result = PipelineResult(
            trace_id=trace.trace_id,
            document_type=final["doc_type"],
            classifier_method=final["classifier_method"],
            fields=final["fields"],
            tables=final.get("tables", []),
            findings=final["findings"],
            overall_confidence=final["confidence"],
            routing=final["routing"],
            pipeline_used=final["pipeline_used"],
            extractor_method=final["extractor_method"],
            vlm_reprocessed=final.get("vlm_reprocessed", False),
            n_pages=final["parsed"].n_pages,
            latency_ms=trace.latency_ms,
            usage=usage,
            node_log=list(trace.node_log),
        )
        logger.bind(trace_id=trace.trace_id).info(
            "pipeline done type={} routing={} conf={} ms={} tokens={}/{} cost={}",
            result.document_type,
            result.routing,
            result.overall_confidence,
            result.latency_ms,
            usage.input_tokens,
            usage.output_tokens,
            usage.cost_usd,
        )
        return result
