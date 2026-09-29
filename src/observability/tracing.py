"""Request tracing: trace ids, node spans, cost accounting and LangSmith wiring."""

from __future__ import annotations

import os
import sys
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.config import Settings
from src.schemas.document import UsageStats


def new_trace_id() -> str:
    """Return a fresh 32-hex-char trace id."""
    return uuid.uuid4().hex


def configure_logging(level: str = "INFO") -> None:
    """Route loguru to stderr with a compact structured format including ``trace_id``.

    Args:
        level: Minimum level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
    """
    logger.remove()
    logger.configure(extra={"trace_id": "-"})
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
            "trace={extra[trace_id]} | {name}:{function} | {message}"
        ),
        backtrace=False,
        diagnose=False,
    )


def configure_langsmith(settings: Settings) -> bool:
    """Export the LangSmith environment variables when tracing is enabled.

    LangGraph picks these up automatically, so no code path changes; without a
    key nothing is exported and nothing is sent.

    Args:
        settings: Active settings.

    Returns:
        True if tracing was enabled.
    """
    if not settings.langsmith_enabled:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    logger.info("LangSmith tracing enabled project={}", settings.langsmith_project)
    return True


def cost_usd(input_tokens: int, output_tokens: int, settings: Settings) -> float:
    """Compute request cost from token counts and configured list prices.

    Args:
        input_tokens: Prompt tokens billed.
        output_tokens: Completion tokens billed.
        settings: Provides ``price_*_per_mtok``.

    Returns:
        Cost in USD rounded to 6 decimals.
    """
    cost = (
        input_tokens * settings.price_input_per_mtok
        + output_tokens * settings.price_output_per_mtok
    ) / 1_000_000
    return round(cost, 6)


@dataclass
class RequestTrace:
    """Mutable accumulator for one pipeline run."""

    trace_id: str = field(default_factory=new_trace_id)
    started: float = field(default_factory=time.perf_counter)
    usage: UsageStats = field(default_factory=UsageStats)
    node_log: list[dict[str, str | int | float]] = field(default_factory=list)

    @property
    def latency_ms(self) -> int:
        """Milliseconds elapsed since the trace started."""
        return int((time.perf_counter() - self.started) * 1000)

    def add_usage(self, usage: UsageStats) -> None:
        """Accumulate token/cost usage from one LLM call."""
        self.usage = self.usage.add(usage)

    @contextmanager
    def span(self, node: str, **inputs: Any) -> Iterator[dict[str, Any]]:
        """Log the entry and exit of a pipeline node and record its duration.

        The caller may put summary values into the yielded dict; they are logged
        as the node output.

        Args:
            node: Node name.
            **inputs: Summary of the node input (already small/serialisable).
        """
        start = time.perf_counter()
        log = logger.bind(trace_id=self.trace_id)
        log.info("node={} status=start input={}", node, inputs)
        outputs: dict[str, Any] = {}
        try:
            yield outputs
        except Exception as exc:
            elapsed = int((time.perf_counter() - start) * 1000)
            log.error("node={} status=error ms={} error={!r}", node, elapsed, exc)
            self.node_log.append(
                {"node": node, "status": "error", "ms": elapsed, "error": repr(exc)}
            )
            raise
        elapsed = int((time.perf_counter() - start) * 1000)
        log.info("node={} status=ok ms={} output={}", node, elapsed, outputs)
        entry: dict[str, str | int | float] = {"node": node, "status": "ok", "ms": elapsed}
        for k, v in outputs.items():
            if isinstance(v, str | int | float):
                entry[k] = v
        self.node_log.append(entry)

    def summary(self) -> dict[str, str | int | float]:
        """Return the per-request observability record (latency, tokens, cost)."""
        return {
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cost_usd": self.usage.cost_usd,
            "llm_calls": self.usage.llm_calls,
        }


class RecentRequests:
    """Fixed-size in-memory ring of the latest request summaries (dashboard feed)."""

    def __init__(self, maxlen: int = 100) -> None:
        self._items: deque[dict[str, Any]] = deque(maxlen=maxlen)

    def record(self, item: dict[str, Any]) -> None:
        """Append one request summary, evicting the oldest when full."""
        self._items.append(item)

    def items(self) -> list[dict[str, Any]]:
        """Return summaries newest first."""
        return list(reversed(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def stats(self) -> dict[str, float | int]:
        """Aggregate latency percentiles and total cost over the ring."""
        lat = sorted(int(i.get("latency_ms", 0)) for i in self._items)
        if not lat:
            return {"count": 0, "p50_ms": 0, "p95_ms": 0, "total_cost_usd": 0.0}
        p50 = lat[len(lat) // 2]
        p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
        return {
            "count": len(lat),
            "p50_ms": p50,
            "p95_ms": p95,
            "total_cost_usd": round(sum(float(i.get("cost_usd", 0.0)) for i in self._items), 6),
        }
