"""The client contract and the behaviour every client shares.

``Client`` is the structural type a caller depends on -- four verbs, each with an async
twin, plus a capability probe. ``_BaseClient`` supplies the parts that do not vary by
provider: structured output steered through the prompt (overridden by a client with native
support), embeddings that refuse by default (Claude has no embeddings API), the ``supports``
probe, and closing the underlying HTTP pools. A concrete client implements only the four
primitives -- ``complete``, ``acomplete``, ``stream``, ``astream`` -- and declares its
``capabilities``.
"""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, Literal, Protocol, Self, runtime_checkable

from credbox import Secret, scrub_exception, scrub_secrets

from thinchat.errors import (
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
    UnsupportedError,
)
from thinchat.structured_output import make_json_instruction, parse_json

__all__ = ["Capability", "Client", "Provider"]


def build_sdk_client(build: Callable[[], Any], *, provider: str) -> Any:
    """Construct a vendor SDK client, converting ANY construction failure into a content-free
    ``ProviderUnavailableError`` raised from a clean frame.

    The vendor constructor takes the revealed API key as an argument, so its own ``__init__``
    frame holds the plaintext key; a construction failure (e.g. a malformed ``ALL_PROXY`` that
    makes httpx reject the transport URL) would otherwise escape with that key live in a
    traceback frame, disclosed by any frame-dumping excepthook (Sentry / cgitb / rich). We
    cannot scrub the SDK's own frame-locals, so we SEVER the chain: capture only the exception's
    type name (never the object, never a bound reference that keeps its traceback alive), let the
    ``except`` block end (dropping the SDK frame), and raise OUTSIDE it -- so ``__cause__`` and
    ``__context__`` are both ``None`` and the key-bearing frame is unreachable."""
    error_name = None
    try:
        return build()
    except Exception as err:   # not BaseException: spare KeyboardInterrupt / asyncio.CancelledError
        error_name = type(err).__name__   # a type name carries no secret; the object is dropped here
    raise ProviderUnavailableError(f"could not build the {provider} client ({error_name})")

# The four providers thinchat speaks to, as a closed type. The public boundary
# (``make_client``) still takes a runtime ``str`` (a config value is not a Literal) and
# validates it; the internal surfaces carry ``Provider`` so a typo is a static error.
Provider = Literal["claude", "openai", "gemini", "ollama"]

# What a client can do. A caller reads ``supports`` (or catches ``UnsupportedError``)
# rather than assuming: Claude, for one, has completion and streaming but no embeddings.
Capability = Literal["completion", "streaming", "structured_output", "embeddings"]


def _retry_after_seconds(err: object) -> float | None:
    """Return a numeric ``Retry-After`` response header as non-negative finite seconds,
    or None when the response has no usable numeric value."""
    response = getattr(err, "response", None)
    headers  = getattr(response, "headers", None)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    raw = getter("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (OverflowError, TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


@runtime_checkable
class Client(Protocol):
    """One LLM client. ``model`` is the chat model it calls; the verbs turn a prompt into
    a completion (whole, streamed, or JSON) and texts into embeddings. Each has an async
    twin for a caller inside an event loop.

    Raises:
        LLMError: a failed API call, or an empty / malformed reply -- for the
            non-streaming verbs (complete/acomplete/parse/aparse/embed/aembed). For
            stream/astream the error surfaces *while iterating*, not at the call, and an
            empty stream is not itself an error.
        RateLimitError: a subclass of LLMError raised when a rate limit (HTTP 429) outlives
            the SDK's own retries; it carries ``retry_after``. An ``except LLMError`` catches
            it too -- catch it by name only to tell a transient limit from a permanent failure.
        UnsupportedError: embed / aembed on a provider with no embeddings API (Claude);
            check ``supports("embeddings")`` first."""

    model: str
    capabilities: frozenset[Capability]

    def supports(self, capability: Capability) -> bool: ...

    def complete(self, prompt: str, *, system: str | None = None) -> str: ...
    async def acomplete(self, prompt: str, *, system: str | None = None) -> str: ...

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]: ...
    def astream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]: ...

    def parse(
        self, prompt: str, schema: dict[str, object], *, system: str | None = None
    ) -> dict[str, object]: ...
    async def aparse(
        self, prompt: str, schema: dict[str, object], *, system: str | None = None
    ) -> dict[str, object]: ...

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]: ...
    async def aembed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]: ...

    def close(self) -> None: ...
    async def aclose(self) -> None: ...

    def __enter__(self) -> Self: ...
    def __exit__(self, *exc: object) -> None: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *exc: object) -> None: ...


class _BaseClient(ABC):
    """Shared behaviour behind the concrete clients. Subclasses implement the four
    primitives and ``_make_aclient`` (which builds the vendor's async SDK client), set
    ``model`` / ``capabilities``, assign ``_client`` (the sync SDK client), and set
    ``_aclient`` to None (the async client is built lazily on first async use); everything
    else is derived here."""

    model:        str
    capabilities: frozenset[Capability]
    _client:      Any   # the vendor's sync SDK client (owns an HTTP connection pool)
    _aclient:     Any   # the vendor's async SDK client, or None until first async use
    _ratelimit_error: type[BaseException]   # the vendor SDK's 429 exception, mapped to RateLimitError
    _provider_label:  str                   # provider name shown in error messages
    _secret_key:      Secret | None          # the key to scrub from errors; None when there is none (ollama)

    def _map_sdk_failure(self, err: Exception, action: str) -> LLMError:
        """Map a caught SDK error to a rate-limit error with any requested wait, or a plain
        API failure. ``action`` names the failed completion, embedding, or stream.

        The returned error holds NO reference to ``err`` -- only a scrubbed string and a numeric
        ``retry_after`` -- so the caller can raise it OUTSIDE its ``except`` block (never
        ``from err``), leaving ``__cause__``/``__context__`` ``None``. That severance is what
        keeps the key off the traceback: the SDK error carries the key in ``err.request.headers``
        (``Authorization`` / ``x-api-key``) and in its own frame-locals, which ``scrub_exception``
        cannot reach -- so we drop the whole chain rather than try to scrub those. ``scrub_secrets``
        still cleans the rendered ``str(err)`` we interpolate (a provider that echoes the key in
        its message text); ``scrub_exception`` on ``err`` is belt-and-suspenders before it is
        dropped. A keyless provider (ollama) has ``_secret_key`` None, so nothing is redacted."""
        secrets = [self._secret_key] if self._secret_key is not None else []
        scrub_exception(err, secrets)
        scrubbed_detail = scrub_secrets(str(err), secrets)
        if isinstance(err, self._ratelimit_error):
            return RateLimitError(
                f"{self._provider_label} {action} rate-limited: {scrubbed_detail}",
                retry_after=_retry_after_seconds(err),
            )
        return LLMError(f"{self._provider_label} {action} failed: {scrubbed_detail}")

    def supports(self, capability: Capability) -> bool:
        """Whether this client offers ``capability`` -- the check to make before calling
        a verb that might raise ``UnsupportedError`` (e.g. ``embed`` on Claude)."""
        return capability in self.capabilities

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    # --- primitives each client implements natively -------------------------------

    @abstractmethod
    def complete(self, prompt: str, *, system: str | None = None) -> str: ...

    @abstractmethod
    async def acomplete(self, prompt: str, *, system: str | None = None) -> str: ...

    @abstractmethod
    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]: ...

    @abstractmethod
    def astream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]: ...

    # --- async SDK client: built on first async use --------------------------------
    # The async verbs go through _get_aclient() rather than touching _aclient, so a
    # sync-only caller never constructs an async pool it then cannot release (close() and
    # __exit__ are sync and reach only _client). A subclass makes its vendor async client
    # in _make_aclient; this caches it after the first async call. (Not synchronised: two
    # threads first-touching an async verb at once could build two -- fine for a client
    # driven from a single event loop, the normal case.)

    @abstractmethod
    def _make_aclient(self) -> Any: ...

    def _get_aclient(self) -> Any:
        if self._aclient is None:
            self._aclient = self._make_aclient()
        return self._aclient

    # --- structured output: default steers through the prompt ----------------------
    # A client with native constrained output (OpenAI's response_format) overrides the
    # ``_text_for_parse`` hooks; the default appends the schema to the system prompt and
    # parses the reply, which works on any client that can follow an instruction.

    def parse(
        self, prompt: str, schema: dict[str, object], *, system: str | None = None
    ) -> dict[str, object]:
        """Return the reply parsed into a JSON object. ``schema`` is supplied to guide (or
        natively constrain) generation; it is not validated locally, so a caller that needs
        strict conformance checks the returned object itself. Raises ``LLMError`` if the
        reply is not a JSON object."""
        return parse_json(self._text_for_parse(prompt, _with_schema(system, schema)))

    async def aparse(
        self, prompt: str, schema: dict[str, object], *, system: str | None = None
    ) -> dict[str, object]:
        """Async twin of ``parse``."""
        return parse_json(await self._atext_for_parse(prompt, _with_schema(system, schema)))

    def _text_for_parse(self, prompt: str, system: str) -> str:
        """How ``parse`` gets its raw text. The default just completes with the schema
        folded into the system prompt; a client with native JSON mode overrides this to
        constrain the reply at the API as well."""
        return self.complete(prompt, system=system)

    async def _atext_for_parse(self, prompt: str, system: str) -> str:
        """Async twin of ``_text_for_parse``."""
        return await self.acomplete(prompt, system=system)

    # --- embeddings: refused unless a client overrides ----------------------------

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Return one embedding vector per input text. Raises ``UnsupportedError`` on a
        client without an embeddings API (Claude)."""
        raise UnsupportedError(f"{type(self).__name__} does not support embeddings")

    async def aembed(
        self, texts: Sequence[str], *, model: str | None = None
    ) -> list[list[float]]:
        """Async twin of ``embed``."""
        raise UnsupportedError(f"{type(self).__name__} does not support embeddings")

    # --- lifecycle: release the underlying HTTP connection pools -------------------

    def close(self) -> None:
        """Close the sync SDK client's connection pool. Use ``with make_client(...) as c:``
        or call this when done, so a server building a client per request does not leak
        connections. If you drove async verbs (``acomplete`` / ``astream`` / ...), release
        with ``aclose()`` or ``async with`` instead -- that also closes the async pool,
        which this sync close cannot."""
        _close(self._client)

    async def aclose(self) -> None:
        """Close both the async and sync SDK clients' pools. The async pool exists only if
        an async verb was used (it is built lazily); the sync pool is always released, even
        if closing the async one raises."""
        try:
            aclient = self._aclient
            if aclient is not None:
                # The async SDK client's close is a coroutine named ``close``
                # (openai/anthropic), occasionally ``aclose`` elsewhere; accept either and
                # await it if it is one.
                aclose = getattr(aclient, "aclose", None) or getattr(aclient, "close", None)
                if callable(aclose):
                    result = aclose()
                    if inspect.isawaitable(result):
                        await result
        finally:
            _close(self._client)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def _with_schema(system: str | None, schema: dict[str, object]) -> str:
    """Fold the JSON-schema instruction into the system prompt: appended when a system
    prompt is given, else it stands alone as the system prompt."""
    instruction = make_json_instruction(schema)
    return f"{system}\n\n{instruction}" if system else instruction


def _close(client: object) -> None:
    """Call an SDK client's ``close`` if it has one (the fakes and some SDKs do not)."""
    close = getattr(client, "close", None)
    if callable(close):
        close()
