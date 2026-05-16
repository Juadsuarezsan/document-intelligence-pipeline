"""FastAPI for Document Intelligence."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv()

from src.api.schemas import ExtractRequest, ExtractResponse
from src.classifier.doc_type import DocTypeClassifier
from src.config import get_settings
from src.extractors.llm_extractor import LLMExtractor
from src.parsers.text_parser import parse_pdf_b64, parse_text
from src.validators.cross_field import routing_decision, validate_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    app.state.classifier = DocTypeClassifier(model=s.anthropic_model, api_key=s.anthropic_api_key)
    app.state.extractor = LLMExtractor(model=s.anthropic_model, api_key=s.anthropic_api_key)
    yield


app = FastAPI(
    title="Document Intelligence Pipeline",
    version="0.5.0",
    description="Layout → text parse → field extraction → validation → confidence routing.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.5.0",
        "stage": "substantive",
        "vlm_fallback": str(s.use_claude_vision_fallback),
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    start = time.perf_counter()
    try:
        if req.pdf_b64:
            parsed = parse_pdf_b64(req.pdf_b64)
            pipeline = "ocr" if parsed.n_pages > 0 and not parsed.text.strip() else "text"
        elif req.text:
            parsed = parse_text(req.text)
            pipeline = "text"
        else:
            raise HTTPException(status_code=400, detail="Must provide `text` or `pdf_b64`")

        doc_type = req.document_type
        if doc_type == "auto":
            doc_type = await app.state.classifier.classify(parsed.text)
            logger.info(f"classified document_type={doc_type}")

        fields = await app.state.extractor.extract(parsed.text, doc_type)

        s = get_settings()
        confidence, findings = validate_all(fields, doc_type)
        routing = routing_decision(
            confidence,
            auto_thr=s.confidence_auto_approve,
            review_thr=s.confidence_human_review,
        )

        return ExtractResponse(
            document_type=doc_type, fields=fields, tables=[],
            overall_confidence=confidence, validation_findings=findings,
            routing=routing, pipeline_used=pipeline,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("extract failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/eval/run")
async def run_eval() -> dict:
    from src.eval.runner import run_eval as _run
    return await _run()
