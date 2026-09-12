"""The OpenAI-compatible client: every verb (sync + async), structured mode, streaming,
error mapping, and the malformed-response paths that must surface as LLMError."""

import types
from typing import Any

import httpx
import pytest

from tests.fakes import (
    FakeOpenAIError,
    FakeOpenAIRateLimitError,
    fake_embeddings,
    install_openai,
)
from thinchat import make_client
from thinchat.errors import LLMError, RateLimitError


def _client(monkeypatch, **kwargs):
    install_openai(monkeypatch, **kwargs)
    return make_client("openai", api_key="k")


# --- completion ---------------------------------------------------------------

def test_complete_returns_the_reply_text(monkeypatch):
    assert _client(monkeypatch, content="hello").complete("hi") == "hello"


async def test_acomplete_returns_the_reply_text(monkeypatch):
    assert await _client(monkeypatch, content="hello").acomplete("hi") == "hello"


def test_complete_returns_a_completion_with_metadata(monkeypatch):
    from thinchat import Completion
    result = _client(monkeypatch, content="hi there", finish_reason="stop",
                     model="gpt-4o-mini-fake").complete("q")
    assert isinstance(result, Completion) and result == "hi there"   # still a str
    assert result.finish_reason == "stop" and result.truncated is False
    assert result.model == "gpt-4o-mini-fake"
    assert result.usage is not None and result.usage.output_tokens == 11


def test_complete_flags_a_truncated_reply(monkeypatch):
    # OpenAI signals a token-cap cutoff with finish_reason="length".
    result = _client(monkeypatch, content="cut off", finish_reason="length").complete("q")
    assert result.truncated is True and result.finish_reason == "length"


def test_sends_system_then_user_messages(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, capture=seen)
    make_client("openai", api_key="k").complete("hello", system="be terse")
    assert seen["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.parametrize("provider", ["openai", "gemini", "ollama"])
def test_openai_compatible_omits_max_tokens_by_default(monkeypatch, provider):
    # Optional for the OpenAI-compatible API, so it is not sent -- the model decides.
    seen: dict[str, object] = {}
    install_openai(monkeypatch, capture=seen)
    make_client(provider, api_key="k").complete("hi")
    assert "max_tokens" not in seen
    assert "max_completion_tokens" not in seen


@pytest.mark.parametrize(
    ("provider", "field"),
    [("openai", "max_completion_tokens"), ("gemini", "max_tokens"), ("ollama", "max_tokens")],
)
def test_max_tokens_uses_the_provider_correct_field(monkeypatch, provider, field):
    # OpenAI's newer (o-series / GPT-5-class) models reject max_tokens and require
    # max_completion_tokens; the gemini/ollama compat layers still take max_tokens. Pin the
    # field name per provider so sending the wrong one (a 400 on current OpenAI models) is caught.
    seen: dict[str, object] = {}
    install_openai(monkeypatch, capture=seen)
    make_client(provider, api_key="k", max_tokens=1000).complete("hi")
    assert seen[field] == 1000
    other = "max_tokens" if field == "max_completion_tokens" else "max_completion_tokens"
    assert other not in seen


def test_make_client_forwards_temperature_and_top_p_to_openai(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, capture=seen)
    make_client("openai", api_key="k", temperature=0.2, top_p=0.9).complete("hi")
    assert seen["temperature"] == 0.2
    assert seen["top_p"] == 0.9


def test_openai_omits_temperature_and_top_p_by_default(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, capture=seen)
    make_client("openai", api_key="k").complete("hi")
    assert "temperature" not in seen
    assert "top_p" not in seen


def test_make_client_forwards_timeout_and_max_retries_to_the_sdk_client(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("openai", api_key="k", timeout=30.0, max_retries=5)
    assert seen["timeout"] == 30.0
    assert seen["max_retries"] == 5


def test_openai_omits_timeout_and_max_retries_by_default(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("openai", api_key="k")
    assert "timeout" not in seen        # unset -> the SDK's own default stands
    assert "max_retries" not in seen


def test_openai_forwards_zero_valued_knobs(monkeypatch):
    # 0.0 temperature / top_p and 0 max_retries are legitimate; the `is not None` gate must
    # forward them, not drop them as falsy.
    request_seen: dict[str, object] = {}
    client_seen: dict[str, object] = {}
    install_openai(monkeypatch, capture=request_seen, client_capture=client_seen)
    make_client("openai", api_key="k", temperature=0.0, top_p=0.0, max_retries=0).complete("hi")
    assert request_seen["temperature"] == 0.0
    assert request_seen["top_p"] == 0.0
    assert client_seen["max_retries"] == 0


async def test_openai_async_client_gets_the_transport_config(monkeypatch):
    # The lazily-built async client must receive the same timeout / max_retries as the sync one.
    seen: dict[str, object] = {}
    install_openai(monkeypatch, content="hi", aclient_capture=seen)
    await make_client("openai", api_key="k", timeout=30.0, max_retries=5).acomplete("hi")
    assert seen["timeout"] == 30.0
    assert seen["max_retries"] == 5


async def test_openai_async_client_omits_transport_config_by_default(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, content="hi", aclient_capture=seen)
    await make_client("openai", api_key="k").acomplete("hi")
    assert "timeout" not in seen
    assert "max_retries" not in seen


@pytest.mark.parametrize("provider", ["openai", "gemini", "ollama"])
def test_make_client_forwards_zero_max_tokens_to_openai_compatible(monkeypatch, provider):
    # An explicit 0 must be forwarded, not dropped by a truthiness check.
    seen: dict[str, object] = {}
    install_openai(monkeypatch, capture=seen)
    make_client(provider, api_key="k", max_tokens=0).complete("hi")
    field = "max_completion_tokens" if provider == "openai" else "max_tokens"
    assert seen[field] == 0


# --- streaming ----------------------------------------------------------------

def test_stream_yields_the_chunks(monkeypatch):
    assert "".join(_client(monkeypatch, chunks=("a", "b", "c")).stream("hi")) == "abc"


async def test_astream_yields_the_chunks(monkeypatch):
    client = _client(monkeypatch, chunks=("a", "b"))
    assert "".join([chunk async for chunk in client.astream("hi")]) == "ab"


def test_stream_skips_chunks_without_text_delta(monkeypatch):
    # role-only openers / finish markers carry no content and must be skipped, not yielded.
    assert "".join(_client(monkeypatch, chunks=("a", None, "", "b")).stream("hi")) == "ab"


def test_stream_maps_a_midstream_sdk_error_to_llm_error(monkeypatch):
    client = _client(monkeypatch, chunks=("a", "b", "c"), stream_error_after=1)
    stream = client.stream("hi")
    assert next(stream) == "a"
    with pytest.raises(LLMError):
        list(stream)   # the SDK raises after the first chunk


async def test_astream_maps_a_midstream_sdk_error_to_llm_error(monkeypatch):
    client = _client(monkeypatch, chunks=("a", "b"), stream_error_after=1)
    with pytest.raises(LLMError):
        [chunk async for chunk in client.astream("hi")]


def test_stream_maps_a_midstream_rate_limit_and_carries_retry_after(monkeypatch):
    error = FakeOpenAIRateLimitError(retry_after=7)
    client = _client(monkeypatch, chunks=("a", "b"), stream_error_after=1,
                     stream_exc=error)
    stream = client.stream("hi")
    assert next(stream) == "a"
    with pytest.raises(RateLimitError) as excinfo:
        list(stream)
    assert excinfo.value.retry_after == 7.0


async def test_astream_maps_a_midstream_rate_limit_and_carries_retry_after(monkeypatch):
    error = FakeOpenAIRateLimitError(retry_after=7)
    client = _client(monkeypatch, chunks=("a", "b"), stream_error_after=1,
                     stream_exc=error)
    with pytest.raises(RateLimitError) as excinfo:
        [chunk async for chunk in client.astream("hi")]
    assert excinfo.value.retry_after == 7.0


def test_stream_maps_a_midstream_transport_error_to_llm_error(monkeypatch):
    # The SDK does NOT wrap a mid-stream connection drop in OpenAIError -- it raises a raw
    # httpx error -- so the streaming catch must include the transport base too.
    client = _client(monkeypatch, chunks=("a", "b"), stream_error_after=1,
                     stream_exc=httpx.ReadError("connection dropped"))
    stream = client.stream("hi")
    assert next(stream) == "a"
    with pytest.raises(LLMError):
        list(stream)


async def test_astream_maps_a_midstream_transport_error_to_llm_error(monkeypatch):
    client = _client(monkeypatch, chunks=("a", "b"), stream_error_after=1,
                     stream_exc=httpx.ReadError("connection dropped"))
    with pytest.raises(LLMError):
        [chunk async for chunk in client.astream("hi")]


def test_stream_does_not_mask_a_non_sdk_error(monkeypatch):
    # A real bug raised mid-stream (not an SDK/transport error) must surface as itself, or
    # the narrow catch is pointless -- this is what proves _stream_errors is not (Exception,).
    client = _client(monkeypatch, chunks=("a", "b"), stream_error_after=1,
                     stream_exc=RuntimeError("our bug"))
    stream = client.stream("hi")
    assert next(stream) == "a"
    with pytest.raises(RuntimeError):
        list(stream)


async def test_astream_does_not_mask_a_non_sdk_error(monkeypatch):
    client = _client(monkeypatch, chunks=("a", "b"), stream_error_after=1,
                     stream_exc=RuntimeError("our bug"))
    with pytest.raises(RuntimeError):
        [chunk async for chunk in client.astream("hi")]


def test_stream_releases_the_connection_on_early_break(monkeypatch):
    sink: list[Any] = []
    install_openai(monkeypatch, chunks=("a", "b", "c"), stream_sink=sink)
    stream: Any = make_client("openai", api_key="k").stream("hi")
    assert next(stream) == "a"
    stream.close()                       # abandon the generator early
    assert sink[0].closed is True        # the `with` released the SDK stream


async def test_astream_releases_the_connection_on_early_break(monkeypatch):
    sink: list[Any] = []
    install_openai(monkeypatch, chunks=("a", "b", "c"), stream_sink=sink)
    agen: Any = make_client("openai", api_key="k").astream("hi")
    assert await agen.__anext__() == "a"
    await agen.aclose()                  # abandon the async generator early
    assert sink[0].closed is True


# --- structured output --------------------------------------------------------

def test_parse_uses_native_json_mode(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, content='{"is_ad": true}', capture=seen)
    result = make_client("openai", api_key="k").parse("x", {"type": "object"})
    assert result == {"is_ad": True}
    assert seen["response_format"] == {"type": "json_object"}   # openai constrains at the API


async def test_aparse_uses_native_json_mode(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, content='{"ok": true}', capture=seen)
    result = await make_client("openai", api_key="k").aparse("x", {"type": "object"})
    assert result == {"ok": True}
    assert seen["response_format"] == {"type": "json_object"}


def test_gemini_parse_uses_native_json_mode(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, content='{"ok": true}', capture=seen)
    make_client("gemini", api_key="k").parse("x", {"type": "object"})
    assert seen["response_format"] == {"type": "json_object"}


def test_ollama_parse_falls_back_to_prompt(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, content='{"ok": true}', capture=seen)
    make_client("ollama").parse("x", {"type": "object"})
    assert "response_format" not in seen                       # no native JSON mode
    messages = seen["messages"]
    assert isinstance(messages, list)
    assert "JSON" in messages[0]["content"]                    # schema folded into the system prompt


@pytest.mark.parametrize(("provider", "uses_native_json"), [("gemini", True), ("ollama", False)])
async def test_aparse_uses_the_provider_json_strategy(monkeypatch, provider, uses_native_json):
    # The async parse path (_atext_for_parse) must route the same way as the sync one:
    # gemini constrains natively, ollama falls back to the prompt-folded schema.
    seen: dict[str, object] = {}
    install_openai(monkeypatch, content='{"ok": true}', capture=seen)
    await make_client(provider, api_key="k").aparse("x", {"type": "object"})
    assert ("response_format" in seen) is uses_native_json
    if not uses_native_json:
        messages = seen["messages"]
        assert isinstance(messages, list)
        assert "JSON" in messages[0]["content"]


# --- embeddings ---------------------------------------------------------------

def test_gemini_embed_uses_the_current_default_model(monkeypatch):
    # text-embedding-004 was shut down 2026-01-14; the gemini embed default must be a live model.
    seen: dict[str, object] = {}
    install_openai(monkeypatch, capture=seen, vectors=([0.1],))
    make_client("gemini", api_key="k").embed(["a"])
    assert seen["model"] == "gemini-embedding-001"


def test_embed_returns_one_vector_per_input(monkeypatch):
    # Exact ==: the vectors are passed straight through with no arithmetic, so no tolerance.
    client = _client(monkeypatch, vectors=([0.1, 0.2], [0.3, 0.4]))
    assert client.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_returns_vectors_in_input_index_order(monkeypatch):
    # Each response item carries its input `index`; embed() restores input order even when
    # the response items arrive out of order.
    response = types.SimpleNamespace(data=[
        types.SimpleNamespace(index=1, embedding=[2.0]),
        types.SimpleNamespace(index=0, embedding=[1.0]),
    ])
    client = _client(monkeypatch, embedding=response)
    assert client.embed(["first", "second"]) == [[1.0], [2.0]]


async def test_aembed_returns_vectors(monkeypatch):
    assert await _client(monkeypatch, vectors=([0.5],)).aembed(["a"]) == [[0.5]]


def test_embed_of_no_inputs_returns_no_vectors_without_calling(monkeypatch):
    # An empty batch is not an API failure: one vector per input, zero inputs -> [].
    assert _client(monkeypatch, vectors=([0.1],)).embed([]) == []


async def test_aembed_of_no_inputs_returns_empty(monkeypatch):
    assert await _client(monkeypatch, vectors=([0.1],)).aembed([]) == []


# --- error mapping (each handler) ---------------------------------------------

def test_complete_maps_an_sdk_error_to_llm_error(monkeypatch):
    with pytest.raises(LLMError):
        _client(monkeypatch, error=FakeOpenAIError("boom")).complete("hi")


def test_complete_maps_a_rate_limit_and_carries_retry_after(monkeypatch):
    error = FakeOpenAIRateLimitError(retry_after=7)
    with pytest.raises(RateLimitError) as excinfo:
        _client(monkeypatch, error=error).complete("hi")
    assert excinfo.value.retry_after == 7.0


def test_rate_limit_without_retry_after_carries_none(monkeypatch):
    with pytest.raises(RateLimitError) as excinfo:
        _client(monkeypatch, error=FakeOpenAIRateLimitError()).complete("hi")
    assert excinfo.value.retry_after is None


def test_an_absurd_retry_after_is_treated_as_no_hint(monkeypatch):
    # A hostile endpoint returning a huge finite Retry-After must not make a caller that honours
    # retry_after sleep effectively forever; beyond the 24h cap it degrades to no hint.
    with pytest.raises(RateLimitError) as excinfo:
        _client(monkeypatch, error=FakeOpenAIRateLimitError(retry_after=1e308)).complete("hi")
    assert excinfo.value.retry_after is None


def test_rate_limit_error_remains_an_llm_error():
    assert issubclass(RateLimitError, LLMError)


async def test_acomplete_maps_a_rate_limit(monkeypatch):
    with pytest.raises(RateLimitError):
        await _client(monkeypatch, error=FakeOpenAIRateLimitError()).acomplete("hi")


async def test_acomplete_maps_an_sdk_error_to_llm_error(monkeypatch):
    with pytest.raises(LLMError):
        await _client(monkeypatch, error=FakeOpenAIError("boom")).acomplete("hi")


def test_embed_maps_an_sdk_error_to_llm_error(monkeypatch):
    with pytest.raises(LLMError):
        _client(monkeypatch, error=FakeOpenAIError("boom")).embed(["a"])


def test_embed_maps_a_rate_limit_and_carries_retry_after(monkeypatch):
    error = FakeOpenAIRateLimitError(retry_after=7)
    with pytest.raises(RateLimitError) as excinfo:
        _client(monkeypatch, error=error).embed(["a"])
    assert excinfo.value.retry_after == 7.0


async def test_aembed_maps_an_sdk_error_to_llm_error(monkeypatch):
    with pytest.raises(LLMError):
        await _client(monkeypatch, error=FakeOpenAIError("boom")).aembed(["a"])


async def test_aembed_maps_a_rate_limit_and_carries_retry_after(monkeypatch):
    error = FakeOpenAIRateLimitError(retry_after=7)
    with pytest.raises(RateLimitError) as excinfo:
        await _client(monkeypatch, error=error).aembed(["a"])
    assert excinfo.value.retry_after == 7.0


def test_a_non_sdk_error_is_not_masked_as_llm_error(monkeypatch):
    # A bug in our own code (not an SDK error) must surface as itself, not a masked LLMError.
    with pytest.raises(RuntimeError):
        _client(monkeypatch, error=RuntimeError("our bug")).complete("hi")


# --- malformed response shapes -> LLMError ------------------------------------

def test_complete_raises_when_choices_are_empty(monkeypatch):
    with pytest.raises(LLMError):
        _client(monkeypatch, completion=types.SimpleNamespace(choices=[])).complete("hi")


def test_complete_raises_when_content_is_missing(monkeypatch):
    empty = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=None))])
    with pytest.raises(LLMError):
        _client(monkeypatch, completion=empty).complete("hi")


def test_complete_raises_when_content_is_whitespace_only(monkeypatch):
    with pytest.raises(LLMError):
        _client(monkeypatch, content="   ").complete("hi")


def test_embed_raises_when_data_is_empty(monkeypatch):
    with pytest.raises(LLMError):
        _client(monkeypatch, vectors=()).embed(["a"])


def test_embed_raises_when_an_item_has_no_vector(monkeypatch):
    with pytest.raises(LLMError):
        _client(monkeypatch, vectors=(None,)).embed(["a"])


def test_embed_rejects_non_numeric_vector_elements(monkeypatch):
    # A non-conforming gateway could JSON-encode the scalars as strings; that must be an LLMError,
    # not a silently wrong-typed list[list[str]] returned under the list[list[float]] hint.
    response = fake_embeddings([(["0.1", "0.2"], 0)])
    with pytest.raises(LLMError):
        _client(monkeypatch, embedding=response).embed(["a"])


def test_complete_raises_llmerror_on_a_non_list_choices(monkeypatch):
    # A malformed response whose `choices` is a truthy non-list (a dict) must degrade to
    # LLMError, not a raw KeyError from indexing choices[0].
    bad = types.SimpleNamespace(choices={"0": "x"})
    with pytest.raises(LLMError):
        _client(monkeypatch, completion=bad).complete("hi")


def test_embed_raises_when_the_count_does_not_match_the_inputs(monkeypatch):
    # Two inputs but one vector back: a short response would otherwise silently drop an input.
    response = fake_embeddings([([1.0], 0)])
    with pytest.raises(LLMError):
        _client(monkeypatch, embedding=response).embed(["a", "b"])


def test_embed_raises_on_a_duplicate_index(monkeypatch):
    # Two items both claiming index 0: sorting alone would pair a vector with the wrong input,
    # so a duplicate (which is not a permutation of range(2)) is rejected.
    response = fake_embeddings([([1.0], 0), ([2.0], 0)])
    with pytest.raises(LLMError):
        _client(monkeypatch, embedding=response).embed(["a", "b"])


def test_embed_raises_on_an_out_of_range_index(monkeypatch):
    # An index outside 0..n-1 cannot map onto the inputs.
    response = fake_embeddings([([1.0], 0), ([2.0], 5)])
    with pytest.raises(LLMError):
        _client(monkeypatch, embedding=response).embed(["a", "b"])


def test_embed_raises_on_a_mixed_indexed_and_unindexed_response(monkeypatch):
    # One item carries an index and the other does not: the order is ambiguous, so it is
    # rejected rather than guessed.
    response = fake_embeddings([([1.0], 0), ([2.0], None)])
    with pytest.raises(LLMError):
        _client(monkeypatch, embedding=response).embed(["a", "b"])


def test_embed_falls_back_to_positional_order_when_indices_are_absent(monkeypatch):
    # A compat endpoint that omits `index` entirely is trusted in positional order, with the
    # count check still guarding against a short or padded response.
    response = fake_embeddings([([1.0], None), ([2.0], None)])
    assert _client(monkeypatch, embedding=response).embed(["a", "b"]) == [[1.0], [2.0]]


# --- lifecycle ----------------------------------------------------------------

def test_context_manager_and_close_release_the_sync_client(monkeypatch):
    calls: list[str] = []
    install_openai(monkeypatch, content="hi", close_calls=calls)
    with make_client("openai", api_key="k") as llm:
        assert llm.complete("q") == "hi"
    assert calls == ["sync"]   # __exit__ -> close() closed the sync client


async def test_client_is_an_async_context_manager(monkeypatch):
    calls: list[str] = []
    install_openai(monkeypatch, content="hi", close_calls=calls)
    async with make_client("openai", api_key="k") as llm:
        assert await llm.acomplete("q") == "hi"
    assert "async" in calls   # aclose() closed the async client via its `close` coroutine


async def test_aclose_closes_both_the_async_and_sync_clients(monkeypatch):
    calls: list[str] = []
    install_openai(monkeypatch, content="hi", close_calls=calls)
    llm = make_client("openai", api_key="k")
    await llm.acomplete("q")            # first async use builds the async client
    await llm.aclose()
    assert set(calls) == {"async", "sync"}


async def test_aclose_without_async_use_never_builds_the_async_client(monkeypatch):
    # The async client is built lazily on first async use, so a sync-only caller never
    # constructs (nor needs to release) an async pool.
    builds: list[str] = []
    calls: list[str] = []
    install_openai(monkeypatch, close_calls=calls, aclient_builds=builds)
    await make_client("openai", api_key="k").aclose()
    assert builds == []          # no async client was ever constructed
    assert calls == ["sync"]     # only the sync pool was released


async def test_async_operations_reuse_one_async_client(monkeypatch):
    # _get_aclient() caches, so repeated async calls build the async SDK client once.
    builds: list[str] = []
    install_openai(monkeypatch, content="hi", aclient_builds=builds)
    client = make_client("openai", api_key="k")
    assert await client.acomplete("first") == "hi"
    assert await client.acomplete("second") == "hi"
    assert builds == ["async"]   # built once on first async use, then reused
