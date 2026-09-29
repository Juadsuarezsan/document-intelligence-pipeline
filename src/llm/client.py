"""Claude client used by the classifier, extractor, Vision parser and judge.

Design:

* One place owns the ``anthropic.AsyncAnthropic`` instance, the request timeout
  and the tenacity retry policy (exponential backoff on connection, timeout,
  rate-limit and 5xx errors; 4xx errors are not retried).
* Every call returns an :class:`LLMResult` with token usage and cost so callers
  can attribute spend per request.
* :meth:`ClaudeClient.enabled` is False without an API key; callers then use
  their deterministic fallback and never touch the network.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Literal

import anthropic
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import Settings
from src.observability.tracing import cost_usd
from src.schemas.document import UsageStats

ImageMediaType = Literal["image/png", "image/jpeg", "image/webp", "image/gif"]

RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMResult(BaseModel):
    """Text completion plus its accounting."""

    text: str
    model: str
    usage: UsageStats
    stop_reason: str | None = None


class LLMDisabledError(RuntimeError):
    """Raised when an LLM-only path is invoked without an API key."""


class LLMOutputError(ValueError):
    """Raised when Claude's output cannot be parsed into the expected JSON shape."""


def image_block(data: bytes, media_type: ImageMediaType = "image/png") -> dict[str, Any]:
    """Build a base64 ``image`` content block for the Messages API.

    Args:
        data: Raw image bytes.
        media_type: MIME type of ``data``.

    Returns:
        Content block dict ready to be placed in a user message.
    """
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


def text_block(text: str) -> dict[str, Any]:
    """Build a ``text`` content block."""
    return {"type": "text", "text": text}


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating markdown fences.

    Args:
        text: Raw completion text.

    Returns:
        The decoded object.

    Raises:
        LLMOutputError: If the text is not a JSON object.
    """
    body = _FENCE_RE.sub("", text.strip()).strip()
    if not body.startswith("{"):
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise LLMOutputError("no JSON object in model output")
        body = body[start : end + 1]
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"invalid JSON from model: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise LLMOutputError("model output is not a JSON object")
    return data


class ClaudeClient:
    """Async Claude Messages client with retries, timeout and cost accounting.

    Args:
        settings: Provides key, model id, timeout, attempts and prices.
        client: Optional pre-built SDK client (tests inject a mock here).
    """

    def __init__(self, settings: Settings, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._settings = settings
        self.model = settings.anthropic_model
        self._client = client
        if self._client is None and settings.anthropic_api_key:
            # Retries are owned by tenacity below, so the SDK's own retries are off.
            self._client = anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.anthropic_timeout_s,
                max_retries=0,
            )

    @property
    def enabled(self) -> bool:
        """True when a client exists (API key configured or client injected)."""
        return self._client is not None

    async def complete(
        self,
        *,
        system: str,
        content: str | list[dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Send one user turn and return the text completion with usage.

        Args:
            system: System prompt.
            content: Either plain text or a list of content blocks (text/image).
            max_tokens: Completion cap.
            temperature: Sampling temperature (0 for extraction).

        Returns:
            The completion and its accounting.

        Raises:
            LLMDisabledError: If no API key/client is configured.
            anthropic.APIError: After retries are exhausted or on non-retryable errors.
        """
        if self._client is None:
            raise LLMDisabledError("ANTHROPIC_API_KEY not configured")
        blocks = [text_block(content)] if isinstance(content, str) else content
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(RETRYABLE_ERRORS),
            stop=stop_after_attempt(self._settings.anthropic_max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                if attempt.retry_state.attempt_number > 1:
                    logger.warning(
                        "claude retry attempt={} model={}",
                        attempt.retry_state.attempt_number,
                        self.model,
                    )
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    messages=[{"role": "user", "content": blocks}],  # type: ignore[typeddict-item]
                )
        text = "".join(getattr(b, "text", "") for b in response.content if b.type == "text")
        usage = UsageStats(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost_usd(
                response.usage.input_tokens, response.usage.output_tokens, self._settings
            ),
            llm_calls=1,
        )
        logger.debug(
            "claude ok model={} in={} out={} cost={}",
            self.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cost_usd,
        )
        return LLMResult(
            text=text, model=response.model, usage=usage, stop_reason=response.stop_reason
        )

    async def complete_json(
        self,
        *,
        system: str,
        content: str | list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> tuple[dict[str, Any], LLMResult]:
        """Call :meth:`complete` and parse the answer as a JSON object.

        Returns:
            The parsed object and the raw result (for accounting).

        Raises:
            LLMOutputError: If the completion is not a JSON object.
        """
        result = await self.complete(system=system, content=content, max_tokens=max_tokens)
        return parse_json_object(result.text), result
