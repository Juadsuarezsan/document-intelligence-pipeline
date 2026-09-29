import os
from contextlib import asynccontextmanager
from typing import Any

import pytest

from src.config import Settings
from src.observability.tracing import RecentRequests, RequestTrace, configure_langsmith, cost_usd
from src.schemas.document import ExtractedField, PipelineResult, UsageStats
from src.storage.repository import NullRepository, PostgresRepository, build_repository


class FakeConn:
    def __init__(self, log: list[tuple[str, Any]]) -> None:
        self.log = log

    async def execute(self, sql: str, params: Any = None) -> "FakeConn":
        self.log.append((sql.strip().split("(")[0].split("\n")[0], params))
        return self

    async def fetchall(self) -> list[tuple[Any, ...]]:
        return [("t1", "2026-01-01", "invoice", "auto_approve", "text", 0.9, 12, 0, 0, 0.0)]


class FakePool:
    def __init__(self) -> None:
        self.log: list[tuple[str, Any]] = []
        self.closed = False

    @asynccontextmanager
    async def connection(self):  # type: ignore[no-untyped-def]
        yield FakeConn(self.log)

    async def close(self) -> None:
        self.closed = True


def _result() -> PipelineResult:
    return PipelineResult(
        trace_id="t1",
        document_type="invoice",
        classifier_method="heuristic",
        fields=[ExtractedField(name="total", value=1.0, confidence=0.9)],
        routing="auto_approve",
        usage=UsageStats(input_tokens=1, output_tokens=2, cost_usd=0.0001, llm_calls=1),
    )


async def test_postgres_repository_with_fake_pool() -> None:
    pool = FakePool()
    repo = PostgresRepository("postgresql://x", pool=pool)
    assert await repo.save(_result())
    assert pool.log[0][0].startswith("CREATE TABLE")
    assert pool.log[1][0].startswith("INSERT INTO processed_documents")
    assert pool.log[1][1][0] == "t1"
    rows = await repo.recent(5)
    assert rows[0]["trace_id"] == "t1" and rows[0]["document_type"] == "invoice"
    assert await repo.save(_result()) and sum(1 for s, _ in pool.log if s.startswith("CREATE")) == 1
    await repo.close()
    assert pool.closed


async def test_null_repository_and_factory() -> None:
    repo = build_repository(None)
    assert isinstance(repo, NullRepository)
    assert await repo.save(_result()) is False and await repo.recent() == []
    await repo.close()
    assert isinstance(build_repository("postgresql://x"), PostgresRepository)


def test_recent_requests_ring_and_stats() -> None:
    ring = RecentRequests(maxlen=3)
    assert ring.stats()["count"] == 0
    for i, ms in enumerate([50, 10, 30, 20]):
        ring.record({"trace_id": str(i), "latency_ms": ms, "cost_usd": 0.001})
    assert len(ring) == 3 and ring.items()[0]["trace_id"] == "3"
    stats = ring.stats()
    assert stats["count"] == 3 and stats["p50_ms"] == 20 and stats["p95_ms"] == 30
    assert stats["total_cost_usd"] == pytest.approx(0.003)


def test_cost_and_trace_span_records_errors() -> None:
    s = Settings(PRICE_INPUT_PER_MTOK=3.0, PRICE_OUTPUT_PER_MTOK=15.0)
    assert cost_usd(1_000_000, 0, s) == 3.0 and cost_usd(0, 1_000_000, s) == 15.0
    trace = RequestTrace()
    with trace.span("n_ok", x=1) as out:
        out["fields"] = 3
    with pytest.raises(ValueError), trace.span("n_bad"):
        raise ValueError("boom")
    assert trace.node_log[0]["fields"] == 3 and trace.node_log[0]["status"] == "ok"
    assert trace.node_log[1]["status"] == "error"
    trace.add_usage(UsageStats(input_tokens=5, output_tokens=1, cost_usd=0.01, llm_calls=1))
    assert trace.summary()["cost_usd"] == 0.01


def test_langsmith_wiring_only_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert configure_langsmith(Settings(LANGSMITH_API_KEY=None)) is False
    assert (
        configure_langsmith(Settings(LANGSMITH_API_KEY="ls-key", LANGCHAIN_TRACING_V2=False))
        is False
    )
    assert (
        configure_langsmith(Settings(LANGSMITH_API_KEY="ls-key", LANGCHAIN_TRACING_V2=True)) is True
    )
    assert os.environ["LANGCHAIN_PROJECT"] == "document-intelligence"
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
