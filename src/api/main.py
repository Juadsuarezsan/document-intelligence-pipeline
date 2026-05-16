"""Document Intelligence Pipeline — placeholder until v0.1.0 build out."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Document Intelligence Pipeline",
    version="0.1.0",
    description="Document Intelligence — IDP with VLM fallback over FUNSD/DocVQA",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.1.0",
        "stage": "scaffolding",
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }

class ExtractRequest(BaseModel):
    file_url: str | None = None
    pdf_b64: str | None = None
    document_type: str = "auto"


class ExtractedField(BaseModel):
    name: str
    value: str | float | None
    confidence: float


class ExtractResponse(BaseModel):
    document_type: str
    fields: list[ExtractedField]
    tables: list[dict] = []
    overall_confidence: float
    routing: str  # auto_approve | human_review | vlm_fallback


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    return ExtractResponse(
        document_type="not_yet_implemented", fields=[], overall_confidence=0.0, routing="human_review",
    )
