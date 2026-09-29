"""FastAPI transport for the Document Intelligence pipeline.

Responsibilities of this module only: request validation (422), rate limiting,
CORS, mapping domain exceptions to HTTP status codes, trace headers and the
observability feed. All document logic lives in :mod:`src.pipeline`.
"""

# No `from __future__ import annotations` on purpose: the slowapi decorator wraps the
# route functions and FastAPI cannot resolve string annotations through the wrapper.
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.schemas import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    MetricsResponse,
    RequestedType,
)
from src.config import get_settings, package_version
from src.observability.tracing import (
    RecentRequests,
    RequestTrace,
    configure_langsmith,
    configure_logging,
)
from src.parsers.errors import ParserError
from src.parsers.ocr_parser import tesseract_available
from src.parsers.text_parser import check_pdf_bytes, decode_pdf_b64
from src.pipeline.graph import DocumentPipeline, build_deps
from src.schemas.document import DocumentType
from src.storage.repository import NullRepository, Repository, build_repository

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the pipeline, repository and request ring once per process."""
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_langsmith(settings)
    app.state.settings = settings
    app.state.pipeline = DocumentPipeline(build_deps(settings))
    app.state.repository = build_repository(settings.database_url)
    app.state.recent = RecentRequests(maxlen=100)
    logger.info(
        "api ready version={} model={} llm_enabled={} ocr={}",
        package_version(),
        settings.anthropic_model,
        settings.llm_enabled,
        tesseract_available(settings.tesseract_cmd),
    )
    try:
        yield
    finally:
        repo: Repository = app.state.repository
        await repo.close()


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    app = FastAPI(
        title="Document Intelligence Pipeline",
        version=package_version(),
        description="PDF/text -> classify -> parse (text | OCR | Claude Vision) -> extract -> "
        "validate -> confidence routing.",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Trace-Id"],
        expose_headers=["X-Trace-Id"],
    )
    _register_routes(app, settings.rate_limit)
    return app


def _register_routes(app: FastAPI, rate_limit: str) -> None:
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        s = app.state.settings
        return HealthResponse(
            status="ok",
            version=package_version(),
            model=s.anthropic_model,
            llm_enabled=s.llm_enabled,
            vlm_fallback=s.use_claude_vision_fallback and s.llm_enabled,
            ocr_available=tesseract_available(s.tesseract_cmd),
            database=not isinstance(app.state.repository, NullRepository),
        )

    @app.post("/api/extract", response_model=ExtractResponse)
    @limiter.limit(rate_limit)
    async def extract(
        request: Request, body: ExtractRequest, response: Response
    ) -> ExtractResponse:
        pdf_bytes: bytes | None = None
        if body.pdf_b64:
            try:
                pdf_bytes = decode_pdf_b64(body.pdf_b64, app.state.settings.max_pdf_bytes)
            except ParserError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return await _run(
            app,
            response,
            text=body.text,
            pdf_bytes=pdf_bytes,
            document_type=body.requested_type,
            force_vlm=body.force_vlm,
        )

    @app.post("/api/extract/upload", response_model=ExtractResponse)
    @limiter.limit(rate_limit)
    async def extract_upload(
        request: Request,
        response: Response,
        file: UploadFile = File(..., description="PDF file"),
        document_type: RequestedType = Form(default="auto"),
        force_vlm: bool = Form(default=False),
    ) -> ExtractResponse:
        raw = await file.read()
        try:
            pdf_bytes = check_pdf_bytes(raw, app.state.settings.max_pdf_bytes)
        except ParserError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        requested: DocumentType | None = None if document_type == "auto" else document_type
        return await _run(
            app,
            response,
            text=None,
            pdf_bytes=pdf_bytes,
            document_type=requested,
            force_vlm=force_vlm,
        )

    @app.get("/api/metrics/recent", response_model=MetricsResponse)
    async def metrics_recent(limit: int = 100) -> MetricsResponse:
        recent: RecentRequests = app.state.recent
        stats = recent.stats()
        return MetricsResponse(
            count=int(stats["count"]),
            p50_ms=int(stats["p50_ms"]),
            p95_ms=int(stats["p95_ms"]),
            total_cost_usd=float(stats["total_cost_usd"]),
            requests=recent.items()[: max(1, min(limit, 100))],
        )


async def _run(
    app: FastAPI,
    response: Response,
    *,
    text: str | None,
    pdf_bytes: bytes | None,
    document_type: DocumentType | None,
    force_vlm: bool,
) -> ExtractResponse:
    """Run the pipeline and translate domain errors into HTTP codes."""
    trace = RequestTrace()
    response.headers["X-Trace-Id"] = trace.trace_id
    pipeline: DocumentPipeline = app.state.pipeline
    try:
        result = await pipeline.run(
            text=text,
            pdf_bytes=pdf_bytes,
            document_type=document_type,
            force_vlm=force_vlm,
            trace=trace,
        )
    except ParserError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    summary: dict[str, Any] = {
        **trace.summary(),
        "document_type": result.document_type,
        "routing": result.routing,
        "pipeline_used": result.pipeline_used,
        "overall_confidence": result.overall_confidence,
    }
    app.state.recent.record(summary)
    repo: Repository = app.state.repository
    try:
        await repo.save(result)
    except Exception as exc:  # storage must never fail the request; logged loudly
        logger.bind(trace_id=trace.trace_id).error("repository save failed: {!r}", exc)
    return ExtractResponse.from_result(result)


app = create_app()
