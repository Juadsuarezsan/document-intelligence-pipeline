"""Document repository backed by a psycopg3 async connection pool.

The API works without a database: :func:`build_repository` returns a
:class:`NullRepository` when ``DATABASE_URL`` is unset. With a URL, the pool
is opened lazily on first use and the schema is created if missing.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from loguru import logger

from src.schemas.document import PipelineResult

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS processed_documents (
    trace_id            TEXT PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    document_type       TEXT NOT NULL,
    routing             TEXT NOT NULL,
    pipeline_used       TEXT NOT NULL,
    overall_confidence  DOUBLE PRECISION NOT NULL,
    latency_ms          INTEGER NOT NULL,
    input_tokens        INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    cost_usd            DOUBLE PRECISION NOT NULL,
    result              JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS processed_documents_created_idx ON processed_documents (created_at DESC);
"""

INSERT_SQL = """
INSERT INTO processed_documents (
    trace_id, document_type, routing, pipeline_used, overall_confidence,
    latency_ms, input_tokens, output_tokens, cost_usd, result
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
ON CONFLICT (trace_id) DO NOTHING
"""

RECENT_SQL = """
SELECT trace_id, created_at, document_type, routing, pipeline_used,
       overall_confidence, latency_ms, input_tokens, output_tokens, cost_usd
FROM processed_documents ORDER BY created_at DESC LIMIT %s
"""


class Repository(Protocol):
    """Storage interface used by the API."""

    async def save(self, result: PipelineResult) -> bool:
        """Persist one result; returns True if stored."""
        ...

    async def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent request summaries."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...


class NullRepository:
    """No-op storage used when no database is configured."""

    async def save(self, result: PipelineResult) -> bool:
        """Discard the result."""
        return False

    async def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Always empty."""
        return []

    async def close(self) -> None:
        """Nothing to close."""
        return None


class PostgresRepository:
    """psycopg3 pool-backed repository.

    Args:
        database_url: libpq connection string.
        min_size: Minimum pooled connections.
        max_size: Maximum pooled connections.
        pool: Pre-built pool (tests inject a fake here).
    """

    def __init__(
        self, database_url: str, *, min_size: int = 1, max_size: int = 8, pool: Any | None = None
    ) -> None:
        self._url = database_url
        self._min, self._max = min_size, max_size
        self._pool = pool
        self._schema_ready = False

    async def _get_pool(self) -> Any:
        if self._pool is None:
            from psycopg_pool import AsyncConnectionPool  # lazy

            self._pool = AsyncConnectionPool(
                self._url, min_size=self._min, max_size=self._max, open=False
            )
            await self._pool.open()
            logger.info("postgres pool opened min={} max={}", self._min, self._max)
        if not self._schema_ready:
            async with self._pool.connection() as conn:
                await conn.execute(SCHEMA_SQL)
            self._schema_ready = True
        return self._pool

    async def save(self, result: PipelineResult) -> bool:
        """Insert the result (idempotent on ``trace_id``)."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            await conn.execute(
                INSERT_SQL,
                (
                    result.trace_id,
                    result.document_type,
                    result.routing,
                    result.pipeline_used,
                    result.overall_confidence,
                    result.latency_ms,
                    result.usage.input_tokens,
                    result.usage.output_tokens,
                    result.usage.cost_usd,
                    json.dumps(result.model_dump(mode="json")),
                ),
            )
        return True

    async def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch the latest request summaries."""
        pool = await self._get_pool()
        async with pool.connection() as conn:
            cur = await conn.execute(RECENT_SQL, (limit,))
            rows = await cur.fetchall()
        keys = [
            "trace_id",
            "created_at",
            "document_type",
            "routing",
            "pipeline_used",
            "overall_confidence",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "cost_usd",
        ]
        return [dict(zip(keys, row, strict=False)) for row in rows]

    async def close(self) -> None:
        """Close the pool if it was opened."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


def build_repository(database_url: str | None) -> Repository:
    """Return a Postgres repository when a URL is configured, else a null one."""
    if database_url:
        return PostgresRepository(database_url)
    return NullRepository()
