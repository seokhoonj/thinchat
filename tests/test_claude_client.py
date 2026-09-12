"""The Claude client: every verb (sync + async), structured fallback, streaming, the
embeddings gap, error mapping, and the empty / no-text-block reply paths."""

import types
from typing import Any

import httpx
import pytest

from tests.fakes import (
    FakeAnthropicError,
    FakeAnthropicRateLimitError,
    anthropic_text_block,
    install_anthropic,
)
from thinchat import make_client
from thinchat.errors import LLMError, RateLimitError, UnsupportedError


def _client(monkeypatch, **kwargs):
    install_anthropic(monkeypatch, **kwargs)
    return make_client("claude", api_key="k")


# --- completion ---------------------------------------------------------------

def test_complete_returns_the_reply_text(monkeypatch):
    assert _client(monkeypatch, content="hi there").complete("q") == "hi there"


def test_repr_names_the_class_and_model(monkeypatch):
    assert repr(_client(monkeypatch)) == "ClaudeClient(model='claude-haiku-4-5-20251001')"


async def test_acomplete_returns_the_reply_text(monkeypatch):
    assert await _client(monkeypatch, content="hi").acomplete("q") == "hi"


def test_complete_returns_a_completion_with_metadata(monkeypatch):
    from thinchat import Completion
    result = _client(monkeypatch, content="hi", stop_reason="end_turn", model="claude-fake").complete("q")
    assert isinstance(result, Completion) and result == "hi"   # still a str
    assert result.finish_reason == "end_turn" and result.truncated is False
    assert result.model == "claude-fake"
    assert result.usage is not None and result.usage.input_tokens == 7


def test_complete_flags_a_truncated_reply(monkeypatch):
    # Anthropic signals a token-cap cutoff with stop_reason="max_tokens".
    result = _client(monkeypatch, content="cut", stop_reason="max_tokens").complete("q")
    assert result.truncated is True and result.finish_reason == "max_tokens"


def test_system_goes_to_its_own_field(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, capture=seen)
    make_client("claude", api_key="k").complete("hello", system="be terse")
    assert seen["system"] == "be terse"
    assert seen["messages"] == [{"role": "user", "content": "hello"}]


def test_claude_sends_max_tokens_by_default(monkeypatch):
    # Anthropic requires max_tokens, so the client always sends one; 4096 is the default.
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, capture=seen)
    make_client("claude", api_key="k").complete("q")
    assert seen["max_tokens"] == 4096


def test_per_call_model_and_extra_reach_the_request(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, capture=seen)
    make_client("claude", api_key="k", model="claude-haiku-4-5-20251001").complete(
        "q", model="claude-opus-4-8", extra={"top_k": 5})
    assert seen["model"] == "claude-opus-4-8"   # per-call model wins
    assert seen["top_k"] == 5                    # provider-specific field passed through


def test_make_client_forwards_max_tokens_to_claude(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, capture=seen)
    make_client("claude", api_key="k", max_tokens=8192).complete("q")
    assert seen["max_tokens"] == 8192


def test_make_client_forwards_temperature_and_top_p_to_claude(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, capture=seen)
    make_client("claude", api_key="k", temperature=0.2, top_p=0.9).complete("q")
    assert seen["temperature"] == 0.2
    assert seen["top_p"] == 0.9


def test_claude_omits_temperature_and_top_p_by_default(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, capture=seen)
    make_client("claude", api_key="k").complete("q")
    assert "temperature" not in seen
    assert "top_p" not in seen


def test_make_client_forwards_timeout_and_max_retries_to_the_sdk_client(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, client_capture=seen)
    make_client("claude", api_key="k", timeout=30.0, max_retries=5)
    assert seen["timeout"] == 30.0
    assert seen["max_retries"] == 5


def test_claude_omits_timeout_and_max_retries_by_default(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, client_capture=seen)
    make_client("claude", api_key="k")
    assert "timeout" not in seen
    assert "max_retries" not in seen


def test_claude_forwards_zero_valued_knobs(monkeypatch):
    request_seen: dict[str, object] = {}
    client_seen: dict[str, object] = {}
    install_anthropic(monkeypatch, capture=request_seen, client_capture=client_seen)
    make_client("claude", api_key="k", temperature=0.0, top_p=0.0, max_retries=0).complete("q")
    assert request_seen["temperature"] == 0.0
    assert request_seen["top_p"] == 0.0
    assert client_seen["max_retries"] == 0


async def test_claude_async_client_gets_the_transport_config(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, content="hi", aclient_capture=seen)
    await make_client("claude", api_key="k", timeout=30.0, max_retries=5).acomplete("q")
    assert seen["timeout"] == 30.0
    assert seen["max_retries"] == 5


async def test_claude_async_client_omits_transport_config_by_default(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, content="hi", aclient_capture=seen)
    await make_client("claude", api_key="k").acomplete("q")
    assert "timeout" not in seen
    assert "max_retries" not in seen


def test_make_client_forwards_zero_max_tokens_to_claude(monkeypatch):
    # An explicit 0 must be forwarded (not coalesced to the default) -- guards the
    # `is not None` fallback against a regression to a falsy check.
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, capture=seen)
    make_client("claude", api_key="k", max_tokens=0).complete("q")
    assert seen["max_tokens"] == 0


def test_complete_concatenates_multiple_text_blocks(monkeypatch):
    client = _client(monkeypatch, blocks=[anthropic_text_block("foo "), anthropic_text_block("bar")])
    assert client.complete("q") == "foo bar"


# --- streaming ----------------------------------------------------------------

def test_stream_yields_the_chunks(monkeypatch):
    assert "".join(_client(monkeypatch, chunks=("x", "y")).stream("q")) == "xy"


async def test_astream_yields_the_chunks(monkeypatch):
    client = _client(monkeypatch, chunks=("x", "y", "z"))
    assert "".join([chunk async for chunk in client.astream("q")]) == "xyz"


def test_stream_maps_a_midstream_sdk_error_to_llm_error(monkeypatch):
    client = _client(monkeypatch, chunks=("x", "y"), stream_error_after=1)
    stream = client.stream("q")
    assert next(stream) == "x"
    with pytest.raises(LLMError):
        list(stream)


async def test_astream_maps_a_midstream_sdk_error_to_llm_error(monkeypatch):
    client = _client(monkeypatch, chunks=("x", "y"), stream_error_after=1)
    with pytest.raises(LLMError):
        [chunk async for chunk in client.astream("q")]


def test_stream_maps_a_rate_limit_and_carries_retry_after(monkeypatch):
    error = FakeAnthropicRateLimitError(retry_after=7)
    client = _client(monkeypatch, chunks=("x", "y"), stream_error_after=1,
                     stream_exc=error)
    stream = client.stream("q")
    assert next(stream) == "x"
    with pytest.raises(RateLimitError) as excinfo:
        list(stream)
    assert excinfo.value.retry_after == 7.0


async def test_astream_maps_a_rate_limit_and_carries_retry_after(monkeypatch):
    error = FakeAnthropicRateLimitError(retry_after=7)
    client = _client(monkeypatch, chunks=("x", "y"), stream_error_after=1,
                     stream_exc=error)
    with pytest.raises(RateLimitError) as excinfo:
        [chunk async for chunk in client.astream("q")]
    assert excinfo.value.retry_after == 7.0


def test_stream_maps_a_midstream_transport_error_to_llm_error(monkeypatch):
    client = _client(monkeypatch, chunks=("x", "y"), stream_error_after=1,
                     stream_exc=httpx.ReadError("connection dropped"))
    stream = client.stream("q")
    assert next(stream) == "x"
    with pytest.raises(LLMError):
        list(stream)


async def test_astream_maps_a_midstream_transport_error_to_llm_error(monkeypatch):
    client = _client(monkeypatch, chunks=("x", "y"), stream_error_after=1,
                     stream_exc=httpx.ReadError("connection dropped"))
    with pytest.raises(LLMError):
        [chunk async for chunk in client.astream("q")]


def test_stream_does_not_mask_a_non_sdk_error(monkeypatch):
    client = _client(monkeypatch, chunks=("x", "y"), stream_error_after=1,
                     stream_exc=RuntimeError("our bug"))
    stream = client.stream("q")
    assert next(stream) == "x"
    with pytest.raises(RuntimeError):
        list(stream)


def test_stream_releases_the_connection_on_early_break(monkeypatch):
    sink: list[Any] = []
    install_anthropic(monkeypatch, chunks=("x", "y", "z"), stream_sink=sink)
    stream: Any = make_client("claude", api_key="k").stream("q")
    assert next(stream) == "x"
    stream.close()
    assert sink[0].closed is True


async def test_astream_releases_the_connection_on_early_break(monkeypatch):
    sink: list[Any] = []
    install_anthropic(monkeypatch, chunks=("x", "y", "z"), stream_sink=sink)
    agen: Any = make_client("claude", api_key="k").astream("q")
    assert await agen.__anext__() == "x"
    await agen.aclose()                  # abandon the async generator early
    assert sink[0].closed is True        # the `async with` released the SDK stream


async def test_astream_does_not_mask_a_non_sdk_error(monkeypatch):
    # A real bug raised mid-stream must surface as itself, or the narrow catch is pointless.
    client = _client(monkeypatch, chunks=("x", "y"), stream_error_after=1,
                     stream_exc=RuntimeError("our bug"))
    with pytest.raises(RuntimeError):
        [chunk async for chunk in client.astream("q")]


async def test_client_is_an_async_context_manager(monkeypatch):
    calls: list[str] = []
    install_anthropic(monkeypatch, content="hi", close_calls=calls)
    async with make_client("claude", api_key="k") as llm:
        assert await llm.acomplete("q") == "hi"
    assert "async" in calls


async def test_aclose_closes_both_the_async_and_sync_clients(monkeypatch):
    calls: list[str] = []
    install_anthropic(monkeypatch, content="hi", close_calls=calls)
    llm = make_client("claude", api_key="k")
    await llm.acomplete("q")            # first async use builds the async client
    await llm.aclose()
    assert set(calls) == {"async", "sync"}


async def test_aclose_without_async_use_never_builds_the_async_client(monkeypatch):
    # The async client is built lazily on first async use, so a sync-only caller never
    # constructs (nor needs to release) an async pool.
    builds: list[str] = []
    calls: list[str] = []
    install_anthropic(monkeypatch, close_calls=calls, aclient_builds=builds)
    await make_client("claude", api_key="k").aclose()
    assert builds == []          # no async client was ever constructed
    assert calls == ["sync"]     # only the sync pool was released


async def test_async_operations_reuse_one_async_client(monkeypatch):
    # _get_aclient() caches, so repeated async calls build the async SDK client once.
    builds: list[str] = []
    install_anthropic(monkeypatch, content="hi", aclient_builds=builds)
    client = make_client("claude", api_key="k")
    assert await client.acomplete("first") == "hi"
    assert await client.acomplete("second") == "hi"
    assert builds == ["async"]   # built once on first async use, then reused


# --- structured output (prompt fallback -- Claude has no native JSON mode here) ---

def test_parse_steers_through_the_prompt(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, content='{"is_ad": false}', capture=seen)
    result = make_client("claude", api_key="k").parse("q", {"type": "object"}, system="policy")
    assert result == {"is_ad": False}
    system = seen["system"]
    assert isinstance(system, str)
    assert "policy" in system and "JSON" in system   # caller system + schema instruction


async def test_aparse_steers_through_the_prompt(monkeypatch):
    client = _client(monkeypatch, content='{"ok": true}')
    assert await client.aparse("q", {"type": "object"}) == {"ok": True}


# --- embeddings gap -----------------------------------------------------------

def test_embeddings_are_unsupported(monkeypatch):
    client = _client(monkeypatch)
    assert not client.supports("embeddings")
    with pytest.raises(UnsupportedError):
        client.embed(["a"])


async def test_async_embeddings_are_unsupported(monkeypatch):
    with pytest.raises(UnsupportedError):
        await _client(monkeypatch).aembed(["a"])


# --- error mapping ------------------------------------------------------------

def test_complete_maps_an_sdk_error_to_llm_error(monkeypatch):
    with pytest.raises(LLMError):
        _client(monkeypatch, error=FakeAnthropicError("down")).complete("q")


def test_complete_maps_a_rate_limit_and_carries_retry_after(monkeypatch):
    error = FakeAnthropicRateLimitError(retry_after=7)
    with pytest.raises(RateLimitError) as excinfo:
        _client(monkeypatch, error=error).complete("q")
    assert excinfo.value.retry_after == 7.0


def test_rate_limit_without_retry_after_carries_none(monkeypatch):
    with pytest.raises(RateLimitError) as excinfo:
        _client(monkeypatch, error=FakeAnthropicRateLimitError()).complete("q")
    assert excinfo.value.retry_after is None


def test_rate_limit_error_remains_an_llm_error():
    assert issubclass(RateLimitError, LLMError)


async def test_acomplete_maps_a_rate_limit(monkeypatch):
    with pytest.raises(RateLimitError):
        await _client(monkeypatch, error=FakeAnthropicRateLimitError()).acomplete("q")


async def test_acomplete_maps_an_sdk_error_to_llm_error(monkeypatch):
    with pytest.raises(LLMError):
        await _client(monkeypatch, error=FakeAnthropicError("down")).acomplete("q")


def test_a_non_sdk_error_is_not_masked_as_llm_error(monkeypatch):
    with pytest.raises(RuntimeError):
        _client(monkeypatch, error=RuntimeError("our bug")).complete("q")


# --- malformed reply shapes -> LLMError ---------------------------------------

def test_complete_raises_when_content_is_empty(monkeypatch):
    with pytest.raises(LLMError):
        _client(monkeypatch, blocks=[]).complete("q")


def test_complete_raises_when_no_text_blocks_exist(monkeypatch):
    non_text = types.SimpleNamespace(type="tool_use", text=None)
    with pytest.raises(LLMError):
        _client(monkeypatch, blocks=[non_text]).complete("q")


def test_complete_raises_when_the_only_text_block_is_whitespace(monkeypatch):
    with pytest.raises(LLMError):
        _client(monkeypatch, blocks=[anthropic_text_block("   ")]).complete("q")


def test_complete_keeps_text_and_ignores_a_non_text_block(monkeypatch):
    blocks = [anthropic_text_block("hello "), types.SimpleNamespace(type="tool_use", text=None)]
    assert _client(monkeypatch, blocks=blocks).complete("q") == "hello"
