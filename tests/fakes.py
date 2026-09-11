"""Fake provider SDKs for the client tests: no network, no keys.

Each installer swaps a fake module into ``sys.modules`` under the SDK's import name, so a
client's lazy ``from openai import ...`` / ``from anthropic import ...`` picks up the fake
when it is constructed. The fakes expose the SDK's error base (``OpenAIError`` /
``AnthropicError``) as ``FakeOpenAIError`` / ``FakeAnthropicError``, so a test simulates an
SDK failure by passing one of those and the client's narrow ``except`` catches it.
``capture`` records the last request's kwargs so a test can assert what was sent.
"""

from __future__ import annotations

import sys
import types


class FakeOpenAIError(Exception):
    """Stands in for ``openai.OpenAIError`` -- the client catches this exact base."""


class FakeAnthropicError(Exception):
    """Stands in for ``anthropic.AnthropicError``."""


class FakeOpenAIRateLimitError(FakeOpenAIError):
    """An OpenAI rate-limit error with an optional numeric ``Retry-After`` header."""

    def __init__(self, message="rate limited", *, retry_after=None):
        super().__init__(message)
        headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
        self.response = types.SimpleNamespace(headers=headers)


class FakeAnthropicRateLimitError(FakeAnthropicError):
    """An Anthropic rate-limit error with an optional numeric ``Retry-After`` header."""

    def __init__(self, message="rate limited", *, retry_after=None):
        super().__init__(message)
        headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
        self.response = types.SimpleNamespace(headers=headers)


async def _aiter(items):
    for item in items:
        yield item


# --- fake openai SDK -----------------------------------------------------------

class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, *, content=None, delta=None):
        self.message = _Message(content)
        self.delta   = _Message(delta)


class _Completion:
    def __init__(self, content):
        self.choices = [_Choice(content=content)]


class _Chunk:
    def __init__(self, delta):
        self.choices = [_Choice(delta=delta)]


class _EmbeddingItem:
    def __init__(self, vector, index):
        self.embedding = vector
        self.index     = index   # the API tags each item with its input position


class _Embeddings:
    def __init__(self, vectors):
        # Real endpoints tag items with their input index (0..n-1, in order); mirror that so
        # the default response exercises the index-alignment path, not just positional order.
        self.data = [_EmbeddingItem(vector, index) for index, vector in enumerate(vectors)]


def fake_embeddings(items):
    """Build an embeddings response from ``(vector, index)`` pairs, for tests that need a
    response the default (0..n-1, in order) does not produce -- a scrambled, duplicated,
    missing, or out-of-range index, or an index-less item (pass ``index=None``). A vector
    aligns to its stated index, so a test can prove misalignment is caught."""
    return types.SimpleNamespace(
        data=[_EmbeddingItem(vector, index) for vector, index in items])


class _SyncStream:
    """A fake openai ``Stream``: a context manager whose iteration yields chunks, optionally
    raising ``exc`` (default ``FakeOpenAIError``; a test may pass a raw ``httpx`` error to
    exercise the transport path) after ``error_after`` chunks. ``closed`` records whether
    the ``with`` block exited (so a test can prove the stream was released)."""

    def __init__(self, chunks, error_after, exc=None):
        self._chunks      = list(chunks)
        self._error_after = error_after
        self._exc         = exc or FakeOpenAIError("stream broke mid-flight")
        self.closed       = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def __iter__(self):
        for i, chunk in enumerate(self._chunks):
            if self._error_after is not None and i == self._error_after:
                raise self._exc
            yield _Chunk(chunk)


class _AsyncStream:
    def __init__(self, chunks, error_after, exc=None):
        self._chunks      = list(chunks)
        self._error_after = error_after
        self._exc         = exc or FakeOpenAIError("stream broke mid-flight")
        self.closed       = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def __aiter__(self):
        for i, chunk in enumerate(self._chunks):
            if self._error_after is not None and i == self._error_after:
                raise self._exc
            yield _Chunk(chunk)


def install_openai(monkeypatch, *, content="hi", chunks=("a", "b"),
                   vectors=None, completion=None, embedding=None,
                   error=None, stream_error_after=None, stream_exc=None, stream_sink=None,
                   close_calls=None, capture=None, client_capture=None, aclient_builds=None,
                   aclient_capture=None, client_error=None):
    """Install a fake ``openai`` module. ``completion`` / ``embedding`` override the whole
    response object (for malformed-shape tests); ``error`` is raised at call time (pass a
    ``FakeOpenAIError`` to exercise error mapping); ``stream_error_after`` raises ``stream_exc``
    (default ``FakeOpenAIError``, or a raw httpx error) mid-stream; ``stream_sink`` collects
    the created stream objects so a test can assert release (``.closed``); ``close_calls``
    records "sync"/"async" as the sync/async client's ``close`` is called; ``client_capture``
    records the ``OpenAI(**kwargs)`` constructor args (base_url, api_key); ``aclient_builds``
    records "async" each time ``AsyncOpenAI`` is constructed (to prove lazy build / caching);
    ``client_error`` is raised from the constructor itself (to exercise the construction path,
    where the revealed key is live in the SDK's ``__init__`` frame)."""
    if vectors is None:
        vectors = ([0.1, 0.2],)   # a nested list would be a shared mutable default in the signature
    module = types.ModuleType("openai")
    module.OpenAIError     = FakeOpenAIError           # type: ignore[attr-defined]
    module.RateLimitError  = FakeOpenAIRateLimitError  # type: ignore[attr-defined]

    def _record(kwargs):
        if capture is not None:
            capture.clear()
            capture.update(kwargs)

    def _open_stream(stream_cls):
        stream = stream_cls(chunks, stream_error_after, stream_exc)
        if stream_sink is not None:
            stream_sink.append(stream)
        return stream

    def _create(**kwargs):
        _record(kwargs)
        if error is not None:
            raise error
        if kwargs.get("stream"):
            return _open_stream(_SyncStream)
        return completion if completion is not None else _Completion(content)

    def _embed(**kwargs):
        _record(kwargs)
        if error is not None:
            raise error
        return embedding if embedding is not None else _Embeddings(vectors)

    async def _acreate(**kwargs):
        _record(kwargs)
        if error is not None:
            raise error
        if kwargs.get("stream"):
            return _open_stream(_AsyncStream)
        return completion if completion is not None else _Completion(content)

    async def _aembed(**kwargs):
        _record(kwargs)
        if error is not None:
            raise error
        return embedding if embedding is not None else _Embeddings(vectors)

    def _sync_close():
        if close_calls is not None:
            close_calls.append("sync")

    async def _async_close():   # the real AsyncOpenAI names its async close `close` (a coroutine)
        if close_calls is not None:
            close_calls.append("async")

    def _client(create, embed, close):
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)),
            embeddings=types.SimpleNamespace(create=embed),
            close=close,
        )

    def _sync_client(**kw):
        if client_capture is not None:
            client_capture.clear()
            client_capture.update(kw)
        if client_error is not None:   # kw holds the revealed api_key -> it is live in this frame
            raise client_error
        return _client(_create, _embed, _sync_close)

    def _async_client(**kw):
        if aclient_builds is not None:
            aclient_builds.append("async")
        if aclient_capture is not None:
            aclient_capture.clear()
            aclient_capture.update(kw)
        if client_error is not None:
            raise client_error
        return _client(_acreate, _aembed, _async_close)

    module.OpenAI      = _sync_client     # type: ignore[attr-defined]
    module.AsyncOpenAI = _async_client    # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)


# --- fake anthropic SDK --------------------------------------------------------

class _TextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _AnthropicMessage:
    def __init__(self, blocks):
        self.content = blocks


class _StreamCtx:
    def __init__(self, chunks, error_after, exc=None):
        self._chunks      = list(chunks)
        self._error_after = error_after
        self._exc         = exc or FakeAnthropicError("stream broke mid-flight")
        self.closed       = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    @property
    def text_stream(self):
        return self._iter()

    def _iter(self):
        for i, chunk in enumerate(self._chunks):
            if self._error_after is not None and i == self._error_after:
                raise self._exc
            yield chunk


class _AsyncStreamCtx:
    def __init__(self, chunks, error_after, exc=None):
        self._chunks      = list(chunks)
        self._error_after = error_after
        self._exc         = exc or FakeAnthropicError("stream broke mid-flight")
        self.closed       = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    @property
    def text_stream(self):
        return self._aiter()

    async def _aiter(self):
        for i, chunk in enumerate(self._chunks):
            if self._error_after is not None and i == self._error_after:
                raise self._exc
            yield chunk


def install_anthropic(monkeypatch, *, content="hi", blocks=None, chunks=("a", "b"),
                      error=None, stream_error_after=None, stream_exc=None, stream_sink=None,
                      close_calls=None, capture=None, client_capture=None, aclient_builds=None,
                      aclient_capture=None, client_error=None):
    """Install a fake ``anthropic`` module. ``blocks`` overrides the response content blocks
    (for empty / no-text-block tests); ``stream_exc`` (default ``FakeAnthropicError``, or a
    raw httpx error) is raised mid-stream; ``stream_sink`` collects the stream objects so a
    test can assert release (``.closed``); ``close_calls`` records "sync"/"async" as each
    client's ``close`` is called; ``client_capture`` records the ``Anthropic(**kwargs)``
    constructor args (base_url, timeout, max_retries); ``aclient_builds`` records "async" each
    time ``AsyncAnthropic`` is constructed (to prove lazy build / caching); ``client_error`` is
    raised from the constructor itself (the revealed key is live in the SDK's ``__init__`` frame)."""
    module = types.ModuleType("anthropic")
    module.AnthropicError = FakeAnthropicError                 # type: ignore[attr-defined]
    module.RateLimitError = FakeAnthropicRateLimitError        # type: ignore[attr-defined]

    def _sync_close():
        if close_calls is not None:
            close_calls.append("sync")

    async def _async_close():
        if close_calls is not None:
            close_calls.append("async")

    def _create(**kwargs):
        if capture is not None:
            capture.clear()
            capture.update(kwargs)
        if error is not None:
            raise error
        content_blocks = blocks if blocks is not None else [_TextBlock(content)]
        return _AnthropicMessage(content_blocks)

    async def _acreate(**kwargs):
        return _create(**kwargs)

    def _open_stream(stream_cls):
        stream = stream_cls(chunks, stream_error_after, stream_exc)
        if stream_sink is not None:
            stream_sink.append(stream)
        return stream

    def _stream(**kwargs):
        if error is not None:
            raise error
        return _open_stream(_StreamCtx)

    def _astream(**kwargs):
        if error is not None:
            raise error
        return _open_stream(_AsyncStreamCtx)

    def _messages(create, stream):
        return types.SimpleNamespace(create=create, stream=stream)

    def _async_anthropic(**kw):
        if aclient_builds is not None:
            aclient_builds.append("async")
        if aclient_capture is not None:
            aclient_capture.clear()
            aclient_capture.update(kw)
        if client_error is not None:
            raise client_error
        return types.SimpleNamespace(messages=_messages(_acreate, _astream), close=_async_close)

    def _sync_anthropic(**kw):
        if client_capture is not None:
            client_capture.clear()
            client_capture.update(kw)
        if client_error is not None:   # kw holds the revealed api_key -> it is live in this frame
            raise client_error
        return types.SimpleNamespace(messages=_messages(_create, _stream), close=_sync_close)

    module.Anthropic      = _sync_anthropic   # type: ignore[attr-defined]
    module.AsyncAnthropic = _async_anthropic                                                                            # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)


def anthropic_text_block(text):
    """A text content block, for tests building custom Claude response shapes."""
    return _TextBlock(text)
