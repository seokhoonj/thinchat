"""The API key never rides out in an LLM error message or through a chained cause.

A provider SDK error carries the key in two places our scrubbing cannot reach: the request
headers (``Authorization`` / ``x-api-key``) and the SDK's own frame-locals. So the failure
mapping does NOT chain the SDK error: it builds an ``LLMError`` from a scrubbed ``str(err)``
holding no reference to the SDK error, and each verb raises it OUTSIDE its ``except`` block,
leaving ``__cause__`` and ``__context__`` both ``None`` -- the key-bearing SDK error is
unreachable from what the caller catches. These provoke each SDK's error paths -- completion,
streaming, embedding, rate-limit, sync and async -- and assert the key is absent from the
message, and that the chain is severed rather than merely scrubbed in place.
"""

import types

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


def _key_bearing_openai_error(message="401 Unauthorized"):
    """A fake OpenAI SDK error carrying the key in its request headers -- the real leak vector
    scrub_exception cannot rewrite, so only severance (dropping the chain) protects it."""
    err = FakeOpenAIError(message)
    err.request = types.SimpleNamespace(headers={"Authorization": f"Bearer {_TEST_API_KEY}"})  # type: ignore[attr-defined]
    return err


def _key_bearing_anthropic_error(message="401 Unauthorized"):
    err = FakeAnthropicError(message)
    err.request = types.SimpleNamespace(headers={"x-api-key": _TEST_API_KEY})  # type: ignore[attr-defined]
    return err


# Every verb builds its LLMError inside an `except` and must `raise` it OUTSIDE, so the
# key-bearing SDK error is never chained. Only openai sync complete had this asserted before; a
# `raise ... from err` regression on any other verb would re-expose the key via request.headers
# while the message-scrub tests stay green. Pin all ten verbs on the chain being severed.

async def test_openai_acomplete_severs_the_key_bearing_headers(monkeypatch):
    install_openai(monkeypatch, error=_key_bearing_openai_error())
    with pytest.raises(LLMError) as exc:
        await make_client("openai", api_key=_TEST_API_KEY).acomplete("hi")
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc.value)


def test_openai_embed_severs_the_key_bearing_headers(monkeypatch):
    install_openai(monkeypatch, error=_key_bearing_openai_error())
    with pytest.raises(LLMError) as exc:
        make_client("openai", api_key=_TEST_API_KEY).embed(["a"])
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc.value)


async def test_openai_aembed_severs_the_key_bearing_headers(monkeypatch):
    install_openai(monkeypatch, error=_key_bearing_openai_error())
    with pytest.raises(LLMError) as exc:
        await make_client("openai", api_key=_TEST_API_KEY).aembed(["a"])
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc.value)


def test_openai_stream_severs_the_key_bearing_headers(monkeypatch):
    install_openai(monkeypatch, stream_error_after=0, stream_exc=_key_bearing_openai_error())
    with pytest.raises(LLMError) as exc:
        list(make_client("openai", api_key=_TEST_API_KEY).stream("hi"))
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc.value)


async def test_openai_astream_severs_the_key_bearing_headers(monkeypatch):
    install_openai(monkeypatch, stream_error_after=0, stream_exc=_key_bearing_openai_error())
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc:
        [chunk async for chunk in client.astream("hi")]
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc.value)


def test_claude_complete_severs_the_key_bearing_headers(monkeypatch):
    install_anthropic(monkeypatch, error=_key_bearing_anthropic_error())
    with pytest.raises(LLMError) as exc:
        make_client("claude", api_key=_TEST_API_KEY).complete("hi")
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc.value)


async def test_claude_acomplete_severs_the_key_bearing_headers(monkeypatch):
    install_anthropic(monkeypatch, error=_key_bearing_anthropic_error())
    with pytest.raises(LLMError) as exc:
        await make_client("claude", api_key=_TEST_API_KEY).acomplete("hi")
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc.value)


def test_claude_stream_severs_the_key_bearing_headers(monkeypatch):
    install_anthropic(monkeypatch, stream_error_after=0, stream_exc=_key_bearing_anthropic_error())
    with pytest.raises(LLMError) as exc:
        list(make_client("claude", api_key=_TEST_API_KEY).stream("hi"))
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc.value)


async def test_claude_astream_severs_the_key_bearing_headers(monkeypatch):
    install_anthropic(monkeypatch, stream_error_after=0, stream_exc=_key_bearing_anthropic_error())
    client = make_client("claude", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc:
        [chunk async for chunk in client.astream("hi")]
    assert exc.value.__cause__ is None and exc.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc.value)


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


def test_the_sdk_error_is_severed_not_chained(monkeypatch):
    # The mapped LLMError must NOT chain the SDK error: chaining would keep the SDK error
    # (and the key in its request headers / frame) reachable from what the caller catches and
    # printable by any traceback. Prove the chain is cut -- __cause__ and __context__ both None.
    sdk_error = FakeOpenAIError(f"transport failed: https://api/x?key={_TEST_API_KEY}")
    install_openai(monkeypatch, error=sdk_error)
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert _TEST_API_KEY not in str(exc_info.value)


def test_a_key_in_the_sdk_error_request_headers_is_unreachable(monkeypatch):
    # The real leak vector: the SDK error carries the key in request.headers['Authorization'],
    # which scrub_exception does NOT rewrite. Severance is what protects it -- with the chain
    # cut, the header-bearing SDK error is simply not reachable from the raised LLMError. Walk
    # the caught error's whole cause/context chain and assert the key appears nowhere in it.
    sdk_error = FakeOpenAIError("401 Unauthorized")
    request = types.SimpleNamespace(headers={"Authorization": f"Bearer {_TEST_API_KEY}"})
    sdk_error.request = request  # type: ignore[attr-defined]  # the SDK hangs the key-bearing request off the error
    install_openai(monkeypatch, error=sdk_error)
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert exc_info.value.__cause__ is None and exc_info.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)
    # and the header on the original SDK error is what would have leaked had it been chained
    assert _TEST_API_KEY in request.headers["Authorization"]


def test_mapping_a_failure_is_total_even_if_the_retry_after_header_raises(monkeypatch):
    # _map_sdk_failure runs while still inside each verb's `except` block. If the Retry-After
    # read let a hostile headers.get() escape, the exception raised there would implicitly chain
    # __context__ to the key-bearing SDK error and defeat the severance. A pathological getter
    # must degrade to retry_after=None and still yield a clean, severed error.
    class _HostileHeaders:
        def get(self, name):
            raise RuntimeError("boom from headers.get")

    err = FakeOpenAIRateLimitError(f"429 for key {_TEST_API_KEY}")
    err.response = types.SimpleNamespace(headers=_HostileHeaders())
    install_openai(monkeypatch, error=err)
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(RateLimitError) as exc_info:
        client.complete("hi")
    assert exc_info.value.retry_after is None   # the raising getter degraded to None, did not escape
    assert exc_info.value.__cause__ is None and exc_info.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


def test_sdk_error_control_characters_are_neutralized_in_the_message(monkeypatch):
    # The SDK error detail comes from the remote service; a hostile endpoint must not smuggle
    # terminal escapes (C0 ESC or C1 CSI) or a carriage return into the LLMError message.
    install_openai(monkeypatch, error=FakeOpenAIError("boom \x1b[31m x \x9b[0m y \r z"))
    client = make_client("openai", api_key="k")
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    message = str(exc_info.value)
    assert "\x1b" not in message and "\x9b" not in message and "\r" not in message
    assert "boom" in message   # the safe text survives


def test_mapping_is_total_even_if_the_status_code_raises(monkeypatch):
    # _map_sdk_failure reads err.status_code inside each verb's except block; a hostile property
    # that raised there would chain __context__ to the key-bearing SDK error. It must degrade to
    # status_code=None and still yield a clean, severed error (the sibling to the retry-after case).
    class _HostileStatusError(FakeOpenAIError):
        @property
        def status_code(self):
            raise RuntimeError("boom from status_code")

    install_openai(monkeypatch, error=_HostileStatusError(f"401 for key {_TEST_API_KEY}"))
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert exc_info.value.status_code is None   # the raising property degraded to None, did not escape
    assert exc_info.value.__cause__ is None and exc_info.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


def test_rate_limit_mapping_is_total_even_if_the_headers_property_raises(monkeypatch):
    # On the 429 path _retry_after_seconds reads err.response.headers; a hostile property that
    # raises there (a bare getattr only swallows AttributeError) would chain __context__ to the
    # key-bearing SDK error. It must degrade to retry_after=None with the chain severed.
    class _RaisingHeaders:
        @property
        def headers(self):
            raise RuntimeError("boom from headers")

    err = FakeOpenAIRateLimitError(f"429 for key {_TEST_API_KEY}")
    err.response = _RaisingHeaders()   # type: ignore[assignment]  # response.headers is a raising property
    install_openai(monkeypatch, error=err)
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(RateLimitError) as exc_info:
        client.complete("hi")
    assert exc_info.value.retry_after is None
    assert exc_info.value.__cause__ is None and exc_info.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


def test_mapping_is_total_even_if_the_error_str_raises(monkeypatch):
    # _map_sdk_failure renders str(err) for the detail; a hostile __str__ must not escape inside
    # the verb's except block (it would chain __context__ to the key-bearing SDK error).
    class _HostileStrError(FakeOpenAIError):
        def __str__(self):
            raise RuntimeError("boom from __str__")

    install_openai(monkeypatch, error=_HostileStrError("unused"))
    client = make_client("openai", api_key=_TEST_API_KEY)
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert exc_info.value.__cause__ is None and exc_info.value.__context__ is None
    _assert_secret_absent(_TEST_API_KEY, exc_info.value)


def test_the_ollama_placeholder_key_is_not_scrubbed(monkeypatch):
    install_openai(monkeypatch, error=FakeOpenAIError("ollama server is not running on localhost"))
    client = make_client("ollama")   # no key -> the dummy placeholder, which is not a secret
    with pytest.raises(LLMError) as exc_info:
        client.complete("hi")
    assert "ollama server is not running on localhost" in str(exc_info.value)
