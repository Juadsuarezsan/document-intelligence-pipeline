"""Shared fixtures: fixed seeds, settings without API key, sample documents, fake Claude."""

from __future__ import annotations

import random
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.config import Settings, get_settings
from src.llm.client import ClaudeClient
from src.synth.pdf import render_document_pdf, render_table_pdf

SEED = 20260516

INVOICE_TEXT = (
    "ACME CONSULTING S.A.S.\n"
    "Bogotá - Calle 100\n"
    "NIT: 900.123.456-8\n"
    "\n"
    "Invoice No: INV-2026-0042\n"
    "Invoice Date: 2026-03-14\n"
    "Due Date: 2026-04-13\n"
    "Bill To: Globex Corporation\n"
    "Payment Terms: Net 30\n"
    "\n"
    "Subtotal: 2,100.00\n"
    "Tax (19%): 399.00\n"
    "Total Due: USD 2,499.00\n"
    "Currency: USD\n"
)

CONTRACT_TEXT = (
    "MASTER SERVICES AGREEMENT\n"
    "Contract No: MSA-2026-001\n"
    "\n"
    "This Agreement is entered into as of March 1, 2026 (the Effective Date) between "
    "Acme Consulting S.A.S. and Globex Corporation.\n"
    "\n"
    "1. Term. The initial term of this Agreement is 24 months from the Effective Date.\n"
    "2. Fees. The total contract value is USD 120,000.00, payable monthly.\n"
    "3. Governing Law: State of Delaware.\n"
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(SEED)


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Never let a developer's real key leak into the tests."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    get_settings.cache_clear()
    from src.api.main import limiter

    limiter.reset()  # the slowapi limiter is process-wide; never let tests bleed into each other
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Settings with no API key and no database."""
    return Settings(ANTHROPIC_API_KEY=None, DATABASE_URL=None)


@pytest.fixture
def invoice_text() -> str:
    return INVOICE_TEXT


@pytest.fixture
def contract_text() -> str:
    return CONTRACT_TEXT


@pytest.fixture
def invoice_pdf() -> bytes:
    """Text-layer PDF of the sample invoice with a ruled line-item table."""
    return render_document_pdf(
        INVOICE_TEXT,
        ["Description", "Qty", "Unit price", "Amount"],
        [
            ["Consulting hours", "10", "120.00", "1200.00"],
            ["Cloud hosting", "1", "900.00", "900.00"],
        ],
        ruled=True,
    )


@pytest.fixture
def table_pdf() -> bytes:
    return render_table_pdf(
        ["Line", "FY2024", "FY2025"],
        [["Revenue", "1,000", "1,200"], ["Opex", "600", "650"]],
        ruled=True,
    )


def make_response(text: str, input_tokens: int = 100, output_tokens: int = 20) -> Any:
    """Build an object shaped like ``anthropic.types.Message``."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        model="claude-sonnet-4-5-20250929",
        stop_reason="end_turn",
    )


class FakeAnthropic:
    """Stand-in for ``anthropic.AsyncAnthropic`` returning canned completions in order."""

    def __init__(self, texts: list[str] | None = None) -> None:
        self.messages = MagicMock()
        self.messages.create = AsyncMock(side_effect=[make_response(t) for t in (texts or [])])

    def queue(self, *texts: str) -> None:
        """Replace the queued completions."""
        self.messages.create = AsyncMock(side_effect=[make_response(t) for t in texts])


@pytest.fixture
def fake_anthropic() -> FakeAnthropic:
    return FakeAnthropic()


@pytest.fixture
def llm(fake_anthropic: FakeAnthropic) -> ClaudeClient:
    """A ClaudeClient wired to the fake SDK client (key not needed)."""
    cfg = Settings(ANTHROPIC_API_KEY="test-key", ANTHROPIC_MAX_ATTEMPTS=3)
    return ClaudeClient(cfg, client=fake_anthropic)  # type: ignore[arg-type]


@pytest.fixture
def client() -> Iterator[TestClient]:
    """API test client with lifespan (heuristic-only configuration)."""
    from src.api.main import create_app

    with TestClient(create_app()) as c:
        yield c
