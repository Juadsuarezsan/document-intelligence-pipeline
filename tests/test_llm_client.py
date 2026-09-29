from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest

from src.config import Settings
from src.llm.client import (
    ClaudeClient,
    LLMDisabledError,
    LLMOutputError,
    image_block,
    parse_json_object,
    text_block,
)
from tests.conftest import FakeAnthropic, make_response


async def test_complete_returns_text_usage_and_cost(
    llm: ClaudeClient, fake_anthropic: FakeAnthropic
) -> None:
    fake_anthropic.queue('{"type": "invoice"}')
    result = await llm.complete(system="sys", content="hello", max_tokens=32)
    assert result.text == '{"type": "invoice"}'
    assert result.usage.input_tokens == 100 and result.usage.output_tokens == 20
    # 100 * 3 / 1e6 + 20 * 15 / 1e6
    assert result.usage.cost_usd == pytest.approx(0.0006)
    assert result.usage.llm_calls == 1
    kwargs = fake_anthropic.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert kwargs["messages"][0]["content"][0] == {"type": "text", "text": "hello"}


async def test_complete_retries_on_rate_limit_then_succeeds(
    llm: ClaudeClient, fake_anthropic: FakeAnthropic, mocker: pytest.MonkeyPatch
) -> None:
    mocker.patch("tenacity.nap.time.sleep")  # type: ignore[attr-defined]
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )
    fake_anthropic.messages.create = AsyncMock(side_effect=[err, make_response("ok")])
    result = await llm.complete(system="s", content="c")
    assert result.text == "ok"
    assert fake_anthropic.messages.create.call_count == 2


async def test_complete_does_not_retry_bad_request(
    llm: ClaudeClient, fake_anthropic: FakeAnthropic
) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.BadRequestError("bad", response=httpx.Response(400, request=request), body=None)
    fake_anthropic.messages.create = AsyncMock(side_effect=[err, make_response("ok")])
    with pytest.raises(anthropic.BadRequestError):
        await llm.complete(system="s", content="c")
    assert fake_anthropic.messages.create.call_count == 1


async def test_complete_exhausts_retries(
    llm: ClaudeClient, fake_anthropic: FakeAnthropic, mocker: pytest.MonkeyPatch
) -> None:
    mocker.patch("tenacity.nap.time.sleep")  # type: ignore[attr-defined]
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    fake_anthropic.messages.create = AsyncMock(side_effect=[err, err, err, make_response("late")])
    with pytest.raises(anthropic.APIConnectionError):
        await llm.complete(system="s", content="c")
    assert fake_anthropic.messages.create.call_count == 3


async def test_disabled_client_raises() -> None:
    client = ClaudeClient(Settings(ANTHROPIC_API_KEY=None))
    assert not client.enabled
    with pytest.raises(LLMDisabledError):
        await client.complete(system="s", content="c")


def test_client_is_built_with_timeout_and_no_sdk_retries() -> None:
    client = ClaudeClient(Settings(ANTHROPIC_API_KEY="k", ANTHROPIC_TIMEOUT_S=7.5))
    assert client.enabled
    assert client._client is not None
    assert client._client.max_retries == 0
    assert client._client.timeout == 7.5


def test_parse_json_object_variants() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": [1, 2]} thanks') == {"a": [1, 2]}
    with pytest.raises(LLMOutputError):
        parse_json_object("no json here")
    with pytest.raises(LLMOutputError):
        parse_json_object("[1, 2]")
    with pytest.raises(LLMOutputError):
        parse_json_object('{"a": }')


async def test_complete_json_and_blocks(llm: ClaudeClient, fake_anthropic: FakeAnthropic) -> None:
    fake_anthropic.queue('```json\n{"x": 1}\n```')
    data, result = await llm.complete_json(
        system="s", content=[image_block(b"\x89PNG"), text_block("t")]
    )
    assert data == {"x": 1}
    assert result.model == "claude-sonnet-4-5-20250929"
    block = image_block(b"abc", "image/jpeg")
    assert block["source"]["media_type"] == "image/jpeg"
    assert block["source"]["data"] == "YWJj"
