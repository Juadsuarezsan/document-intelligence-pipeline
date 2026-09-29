import base64

import pytest
from fastapi.testclient import TestClient

from src.config import get_settings


def test_extract_text_returns_fields_and_trace(client: TestClient, invoice_text: str) -> None:
    r = client.post("/api/extract", json={"text": invoice_text})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["document_type"] == "invoice"
    assert body["pipeline_used"] == "text"
    assert any(f["name"] == "invoice_number" for f in body["fields"])
    assert body["usage"]["cost_usd"] == 0.0
    assert r.headers["X-Trace-Id"] == body["trace_id"]
    assert body["node_log"][0]["node"] == "n_ingest"


def test_extract_pdf_b64(client: TestClient, invoice_pdf: bytes) -> None:
    r = client.post(
        "/api/extract",
        json={"pdf_b64": base64.b64encode(invoice_pdf).decode(), "document_type": "invoice"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["classifier_method"] == "requested" and body["n_pages"] == 1
    assert len(body["tables"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},  # edge: empty input
        {"text": "   "},  # edge: blank text
        {"text": "a", "pdf_b64": "YQ=="},  # both inputs
        {"pdf_b64": "!!not-base64!!"},  # edge: malformed base64
        {"pdf_b64": base64.b64encode(b"hello").decode()},  # not a PDF
        {"text": "x", "document_type": "spreadsheet"},  # unknown type
    ],
)
def test_extract_validation_errors_return_422(client: TestClient, payload: dict[str, str]) -> None:
    r = client.post("/api/extract", json=payload)
    assert r.status_code == 422, r.text


def test_extract_oversized_inputs_return_422(
    monkeypatch: pytest.MonkeyPatch, invoice_pdf: bytes
) -> None:
    monkeypatch.setenv("MAX_TEXT_CHARS", "50")
    monkeypatch.setenv("MAX_PDF_BYTES", str(len(invoice_pdf) - 10))
    get_settings.cache_clear()
    from src.api.main import create_app

    with TestClient(create_app()) as c:
        assert c.post("/api/extract", json={"text": "x" * 51}).status_code == 422
        r = c.post("/api/extract", json={"pdf_b64": base64.b64encode(invoice_pdf).decode()})
        assert r.status_code == 422 and "exceeds" in r.json()["detail"]


def test_upload_endpoint(client: TestClient, invoice_pdf: bytes) -> None:
    r = client.post(
        "/api/extract/upload",
        files={"file": ("invoice.pdf", invoice_pdf, "application/pdf")},
        data={"document_type": "auto"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["document_type"] == "invoice"
    bad = client.post("/api/extract/upload", files={"file": ("x.txt", b"plain", "text/plain")})
    assert bad.status_code == 422


def test_metrics_recent_reports_requests(client: TestClient, invoice_text: str) -> None:
    client.post("/api/extract", json={"text": invoice_text})
    r = client.get("/api/metrics/recent")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1 and body["requests"][0]["document_type"] == "invoice"
    assert {"trace_id", "latency_ms", "input_tokens", "output_tokens", "cost_usd"} <= set(
        body["requests"][0]
    )


def test_rate_limit_enforced(monkeypatch: pytest.MonkeyPatch, invoice_text: str) -> None:
    monkeypatch.setenv("RATE_LIMIT", "2/minute")
    get_settings.cache_clear()
    from src.api.main import create_app, limiter

    limiter.reset()
    with TestClient(create_app()) as c:
        codes = [c.post("/api/extract", json={"text": invoice_text}).status_code for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429
    limiter.reset()


def test_cors_origins_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://demo.example")
    get_settings.cache_clear()
    from src.api.main import create_app

    with TestClient(create_app()) as c:
        r = c.options(
            "/api/extract",
            headers={"Origin": "https://demo.example", "Access-Control-Request-Method": "POST"},
        )
        assert r.headers.get("access-control-allow-origin") == "https://demo.example"
        r2 = c.options(
            "/api/extract",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in r2.headers
