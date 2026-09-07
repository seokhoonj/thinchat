"""The API key never rides out in an LLM error message or its chained cause.

A provider SDK error can embed the request -- and a provider that puts the key in a URL would
put it in the error string; our failure mapping interpolates ``str(err)`` and re-raises
``from err``, so without scrubbing a caller that logs the exception would print the key. These
provoke each SDK's error paths -- completion, streaming, embedding, rate-limit, sync and async
-- and assert the key is absent from the message and the whole cause chain.
"""

import pytest

from tests.fakes import (
    FakeAnthropicError,
    FakeAnthropicRateLimitError,
    FakeOpenAIError,
    FakeOpenAIRateLimitError,
    install_anthropic,
    install_openai,
)
from thinchat import make_client
from thinchat.errors import LLMError, RateLimitError

_TEST_API_KEY = "sk-super-secret-KEY-should-never-leak-42"


def _assert_secret_absent(secret, error):
    """Assert ``secret`` appears nowhere in ``error`` or its ``__cause__`` / ``__context__``
    chain -- a printed traceback follows both links."""
    seen: set[int] = set()
    stack = [error]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        assert secret not in str(node)
        stack.extend((node.__cause__, node.__context__))


def test_an_openai_error_message_scrubs_the_key(capsys, monkeypatch):
    install_openai(monkeypatch, error=FakeOpenAIError(f"401 auth failed for key {_TEST_API_KEY}"))
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    message = str(exc_info.value)
    assert _TEST_API_KEY not in message
    assert "401 auth failed for key" in message and "***" in message   # safe text and mask survive
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


def test_an_openai_rate_limit_scrubs_the_key_without_losing_retry_after(monkeypatch):
    install_openai(monkeypatch, error=FakeOpenAIRateLimitError(f"429 for key {_TEST_API_KEY}", retry_after=3))
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(RateLimitError) as exc_info:
        client.complete("hi")
    assert _TEST_API_KEY not in str(exc_info.value)
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)
    assert exc_info.value.retry_after == 3   # scrubbing must not disturb the retry hint


def test_an_openai_stream_error_scrubs_the_key(monkeypatch):
    install_openai(monkeypatch, stream_error_after=0,
                   stream_exc=FakeOpenAIError(f"stream broke for key {_TEST_API_KEY}"))
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        list(client.stream("hi"))
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


def test_an_openai_embedding_error_scrubs_the_key(monkeypatch):
    install_openai(monkeypatch, error=FakeOpenAIError(f"embed failed for key {_TEST_API_KEY}"))
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.embed(["text"])
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


async def test_an_openai_async_error_scrubs_the_key(monkeypatch):
    install_openai(monkeypatch, error=FakeOpenAIError(f"async 401 for key {_TEST_API_KEY}"))
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        await client.acomplete("hi")
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


def test_an_anthropic_error_message_scrubs_the_key(monkeypatch):
    install_anthropic(monkeypatch, error=FakeAnthropicError(f"401 auth failed for key {_TEST_API_KEY}"))
    client = make_client("claude", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert _TEST_API_KEY not in str(exc_info.value)
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


def test_an_anthropic_rate_limit_scrubs_the_key_without_losing_retry_after(monkeypatch):
    install_anthropic(monkeypatch,
                      error=FakeAnthropicRateLimitError(f"429 for key {_TEST_API_KEY}", retry_after=5))
    client = make_client("claude", api_key=_TEST_API_KEY)
    with pytest.raises(RateLimitError) as exc_info:
        client.complete("hi")
    assert _TEST_API_KEY not in str(exc_info.value)
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)
    assert exc_info.value.retry_after == 5


def test_an_anthropic_stream_error_scrubs_the_key(monkeypatch):
    install_anthropic(monkeypatch, stream_error_after=0,
                      stream_exc=FakeAnthropicError(f"stream broke for key {_TEST_API_KEY}"))
    client = make_client("claude", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        list(client.stream("hi"))
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


def test_the_chained_cause_is_scrubbed_in_place(monkeypatch):
    sdk_error = FakeOpenAIError(f"transport failed: https://api/x?key={_TEST_API_KEY}")
    install_openai(monkeypatch, error=sdk_error)
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert exc_info.value.__cause__ is sdk_error
    assert _TEST_API_KEY not in str(sdk_error)


def test_the_ollama_placeholder_key_is_not_scrubbed(monkeypatch):
    install_openai(monkeypatch, error=FakeOpenAIError("ollama server is not running on localhost"))
    client = make_client("ollama")   # no key -> the dummy placeholder, which is not a secret
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert "ollama server is not running on localhost" in str(exc_info.value)
