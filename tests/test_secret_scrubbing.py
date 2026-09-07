"""The API key never rides out in an LLM error message or its chained cause.

A provider SDK error can embed the request -- and a provider that puts the key in a URL would
put it in the error string; our failure mapping interpolates ``str(err)`` and re-raises
``from err``, so without scrubbing a caller that logs the exception would print the key. These
provoke each SDK's error paths and assert the key is absent from the message and the whole
cause chain (PACKAGE_BOUNDARY Ch 12).
"""

import pytest

from tests.fakes import (
    FakeAnthropicError,
    FakeOpenAIError,
    FakeOpenAIRateLimitError,
    install_anthropic,
    install_openai,
)
from thinchat import make_client
from thinchat.errors import LLMError, RateLimitError

_KEY = "sk-super-secret-KEY-should-never-leak-42"


def _assert_absent_from_chain(secret, error):
    while error is not None:
        assert secret not in str(error)
        error = error.__cause__


def test_an_openai_error_message_scrubs_the_key(monkeypatch):
    install_openai(monkeypatch, error=FakeOpenAIError(f"401 auth failed for key {_KEY}"))
    client = make_client("openai", api_key=_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert _KEY not in str(exc_info.value)
    _assert_absent_from_chain(_KEY, exc_info.value)


def test_an_openai_rate_limit_scrubs_the_key_without_losing_retry_after(monkeypatch):
    install_openai(monkeypatch, error=FakeOpenAIRateLimitError(f"429 for key {_KEY}", retry_after=3))
    client = make_client("openai", api_key=_KEY)
    with pytest.raises(RateLimitError) as exc_info:
        client.complete("hi")
    assert _KEY not in str(exc_info.value)
    _assert_absent_from_chain(_KEY, exc_info.value)
    assert exc_info.value.retry_after == 3   # scrubbing must not disturb the retry hint


def test_an_anthropic_error_message_scrubs_the_key(monkeypatch):
    install_anthropic(monkeypatch, error=FakeAnthropicError(f"401 auth failed for key {_KEY}"))
    client = make_client("claude", api_key=_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert _KEY not in str(exc_info.value)
    _assert_absent_from_chain(_KEY, exc_info.value)


def test_the_chained_cause_is_scrubbed_in_place(monkeypatch):
    # A caller doing logging.exception prints the whole chain, so the raw SDK error (our
    # __cause__) must be scrubbed too, not only the message we build.
    sdk_error = FakeOpenAIError(f"transport failed: https://api/x?key={_KEY}")
    install_openai(monkeypatch, error=sdk_error)
    client = make_client("openai", api_key=_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert exc_info.value.__cause__ is sdk_error
    assert _KEY not in str(sdk_error)   # the cause was redacted in place


def test_the_ollama_placeholder_key_is_not_scrubbed(monkeypatch):
    # ollama's dummy key "ollama" is not a secret; redacting it would mangle unrelated text.
    install_openai(monkeypatch, error=FakeOpenAIError("ollama server is not running on localhost"))
    client = make_client("ollama")   # no key -> the dummy placeholder
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert "ollama server is not running on localhost" in str(exc_info.value)
