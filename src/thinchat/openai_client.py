"""The three clients that speak the OpenAI-compatible API: openai, gemini, ollama.

OpenAI, Gemini, and Ollama all expose the same ``/chat/completions`` and ``/embeddings``
shapes, so one client over the ``openai`` SDK serves all three -- they differ only in
data: the base URL, the key, the default models, and whether the endpoint honours a
native JSON mode. Gemini and Ollama are reached by pointing the SDK's ``base_url`` at
their compatible endpoints; only the provider rows below change.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import thinchat.keys as keys
from thinchat.client import Capability, _BaseClient
from thinchat.errors import LLMError, ProviderUnavailableError, UnknownProviderError

__all__ = ["OpenAICompatibleClient"]

# An OpenAI-compatible client does everything; only Claude drops a capability.
_CAPABILITIES: frozenset[Capability] = frozenset(
    {"completion", "streaming", "structured_output", "embeddings"}
)


@dataclass(frozen=True, kw_only=True)
class _ProviderSpec:
    """The data that distinguishes one OpenAI-compatible provider from another. ``is_local``
    is True for Ollama -- a local server that authenticates nothing and whose ``base_url``
    is resolved from ``OLLAMA_HOST`` at construction (so the ``base_url`` field is ignored).
    ``has_native_json`` marks an endpoint that honours ``response_format`` -- OpenAI and
    Gemini do; Ollama's compat layer does not, so it falls back to prompt-steered JSON.
    Keyword-only so two same-typed model ids cannot be transposed by position."""

    base_url:        str | None
    chat_model:      str
    embed_model:     str
    has_native_json: bool
    needs_key:       bool
    is_local:        bool


# The three providers, as data. A `-latest` alias for Gemini, not a pinned name: Google
# gates pinned model ids (e.g. gemini-2.5-flash-lite) to 404 for newly created API keys,
# while the alias tracks the current model and stays on the free tier. Private: reached
# only through _make_openai_client, not exported.
_SPEC_BY_PROVIDER: dict[str, _ProviderSpec] = {
    "openai": _ProviderSpec(
        base_url=None, chat_model="gpt-4o-mini", embed_model="text-embedding-3-small",
        has_native_json=True, needs_key=True, is_local=False),
    "gemini": _ProviderSpec(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        chat_model="gemini-flash-lite-latest", embed_model="text-embedding-004",
        has_native_json=True, needs_key=True, is_local=False),
    "ollama": _ProviderSpec(
        base_url=None, chat_model="llama3.1", embed_model="nomic-embed-text",
        has_native_json=False, needs_key=False, is_local=True),
}


class OpenAICompatibleClient(_BaseClient):
    """A client over the ``openai`` SDK, pointed at any OpenAI-compatible endpoint. Holds a
    sync client and lazily builds a reusable async client on first async use, plus the chat
    and embedding model ids, so the same class serves OpenAI, Gemini, and Ollama."""

    _sdk_error:     type[Exception]
    _stream_errors: tuple[type[Exception], ...]

    def __init__(
        self, *, provider: str, base_url: str | None, api_key: str, model: str,
        embed_model: str, has_native_json: bool, max_tokens: int | None = None,
        temperature: float | None = None, top_p: float | None = None,
        timeout: float | None = None, max_retries: int | None = None,
    ) -> None:
        try:
            import httpx
            from openai import OpenAI, OpenAIError
            from openai import RateLimitError as _OpenAIRateLimitError
        except ImportError as err:
            raise ProviderUnavailableError(
                "the openai package is required but could not be imported; reinstall thinchat"
            ) from err
        self.model            = model
        self.capabilities     = _CAPABILITIES
        self._provider        = provider
        self._base_url        = base_url
        self._api_key         = api_key
        self._embed_model     = embed_model
        self._has_native_json = has_native_json
        self._max_tokens      = max_tokens    # None -> omit (OpenAI-compatible; the model decides)
        self._temperature     = temperature   # None -> omit (sampling knobs go in the request)
        self._top_p           = top_p
        self._timeout         = timeout        # None -> the SDK default (transport, on the client)
        self._max_retries     = max_retries
        # Catch only the SDK's own error family so a bug in our code surfaces as itself.
        self._sdk_error       = OpenAIError
        self._ratelimit_error = _OpenAIRateLimitError
        self._provider_label  = provider
        # The streaming iteration path does NOT wrap transport failures in OpenAIError, so
        # a mid-stream disconnect raises a raw httpx error; catch that base there too.
        self._stream_errors   = (OpenAIError, httpx.HTTPError)
        self._client          = OpenAI(base_url=base_url, api_key=api_key, **self._transport_kwargs())
        self._aclient         = None   # built on first async use (see _make_aclient)

    def _transport_kwargs(self) -> dict[str, Any]:
        # timeout / max_retries are HTTP-client config for the SDK constructor; send each
        # only when set so the SDK's own default stands otherwise. dict[str, Any]: the bag is
        # **-splatted into the vendor constructor, whose kwargs are heterogeneously typed, so
        # a tighter value type won't unpack cleanly under mypy --strict.
        kwargs: dict[str, Any] = {}
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        if self._max_retries is not None:
            kwargs["max_retries"] = self._max_retries
        return kwargs

    def _make_aclient(self) -> Any:
        from openai import AsyncOpenAI  # the sync import above already proved it installed
        return AsyncOpenAI(base_url=self._base_url, api_key=self._api_key, **self._transport_kwargs())

    def __repr__(self) -> str:   # one class serves three providers; show which
        return f"OpenAICompatibleClient(provider={self._provider!r}, model={self.model!r})"

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        return self._chat(self._make_messages(prompt, system))

    async def acomplete(self, prompt: str, *, system: str | None = None) -> str:
        return await self._achat(self._make_messages(prompt, system))

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        request = self._make_request(self._make_messages(prompt, system), json_mode=False)
        request["stream"] = True
        try:
            # The SDK stream owns the HTTP connection and is a context manager; `with`
            # releases it on an early break or a consumer exception. `Any`: the streamed
            # type is hidden behind **request, so chunks are read defensively via
            # _extract_stream_text.
            sdk_stream: Any = self._client.chat.completions.create(**request)
            with sdk_stream as events:
                for chunk in events:
                    delta = _extract_stream_text(chunk)
                    if delta:
                        yield delta
        except self._stream_errors as err:
            raise self._sdk_failure(err, "stream") from err

    async def astream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        request = self._make_request(self._make_messages(prompt, system), json_mode=False)
        request["stream"] = True
        try:
            sdk_stream: Any = await self._get_aclient().chat.completions.create(**request)
            async with sdk_stream as events:
                async for chunk in events:
                    delta = _extract_stream_text(chunk)
                    if delta:
                        yield delta
        except self._stream_errors as err:
            raise self._sdk_failure(err, "stream") from err

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Return one embedding vector per input text (empty input -> empty list, no call).
        Raises ``LLMError`` if the API call fails or a reply carries no vector."""
        text_list = list(texts)
        if not text_list:
            return []   # one vector per input; zero inputs -> zero vectors, not an error
        try:
            response = self._client.embeddings.create(model=model or self._embed_model, input=text_list)
        except self._sdk_error as err:
            raise self._sdk_failure(err, "embedding") from err
        return _extract_embedding_vectors(response)

    async def aembed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Async twin of ``embed``."""
        text_list = list(texts)
        if not text_list:
            return []
        try:
            response = await self._get_aclient().embeddings.create(model=model or self._embed_model, input=text_list)
        except self._sdk_error as err:
            raise self._sdk_failure(err, "embedding") from err
        return _extract_embedding_vectors(response)

    # Native JSON mode where the endpoint honours it, else the base's prompt-steered path.
    def _text_for_parse(self, prompt: str, system: str) -> str:
        return self._chat(self._make_messages(prompt, system), json_mode=self._has_native_json)

    async def _atext_for_parse(self, prompt: str, system: str) -> str:
        return await self._achat(self._make_messages(prompt, system), json_mode=self._has_native_json)

    def _chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            response = self._client.chat.completions.create(**self._make_request(messages, json_mode=json_mode))
        except self._sdk_error as err:
            raise self._sdk_failure(err, "completion") from err
        return _extract_chat_text(response)

    async def _achat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            response = await self._get_aclient().chat.completions.create(**self._make_request(messages, json_mode=json_mode))
        except self._sdk_error as err:
            raise self._sdk_failure(err, "completion") from err
        return _extract_chat_text(response)

    def _make_request(self, messages: list[dict[str, str]], *, json_mode: bool) -> dict[str, object]:
        request: dict[str, object] = {"model": self.model, "messages": messages}
        if self._max_tokens is not None:   # optional here, so send it only when set
            request["max_tokens"] = self._max_tokens
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._top_p is not None:
            request["top_p"] = self._top_p
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        return request

    def _make_messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages


def _make_openai_client(
    provider: str, *, model: str | None = None, api_key: str | None = None,
    max_tokens: int | None = None, temperature: float | None = None,
    top_p: float | None = None, timeout: float | None = None,
    max_retries: int | None = None,
) -> OpenAICompatibleClient:
    """Construct one of the OpenAI-compatible clients (openai / gemini / ollama) by name.
    Each setting is sent only when set; when None the provider's own default stands.

    Raises:
        UnknownProviderError: ``provider`` is not an OpenAI-compatible provider.
        ProviderUnavailableError: the openai SDK is not installed, or the provider needs a
            key and none is set (or passed).
    """
    spec = _SPEC_BY_PROVIDER.get(provider)
    if spec is None:
        raise UnknownProviderError(
            f"{provider!r} is not an OpenAI-compatible provider; "
            f"choose one of {', '.join(_SPEC_BY_PROVIDER)}"
        )
    key = keys.get_api_key(provider, override=api_key)
    if spec.needs_key and not key:
        raise ProviderUnavailableError(
            f"no API key for {provider}: pass api_key=, set {keys.ENV_BY_PROVIDER[provider]}, "
            f"or run 'thinchat set {provider}'"
        )
    base_url = _ollama_base_url() if spec.is_local else spec.base_url
    return OpenAICompatibleClient(
        provider        = provider,
        base_url        = base_url,
        api_key         = key or "ollama",   # Ollama ignores the key, but the SDK needs a non-empty one
        model           = model or spec.chat_model,
        embed_model     = spec.embed_model,
        has_native_json = spec.has_native_json,
        max_tokens      = max_tokens,
        temperature     = temperature,
        top_p           = top_p,
        timeout         = timeout,
        max_retries     = max_retries,
    )


def _ollama_base_url() -> str:
    """Ollama's OpenAI-compatible endpoint, from ``OLLAMA_HOST`` or the local default. Adds
    a scheme (Ollama accepts a bare ``host:port`` that httpx cannot use) and the ``/v1``
    suffix the SDK expects."""
    raw  = os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    host = (raw if "://" in raw else f"http://{raw}").rstrip("/")
    return host if host.endswith("/v1") else host + "/v1"


def _extract_chat_text(response: object) -> str:
    """The first message's text from a chat-completions response, or an ``LLMError`` when
    the reply carries none (an empty choice list or empty content)."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMError("completion returned no choices")
    text = getattr(getattr(choices[0], "message", None), "content", None)
    if not isinstance(text, str) or not text.strip():   # blank reply is empty, like Claude's
        raise LLMError("completion returned an empty reply")
    return text


def _extract_stream_text(chunk: object) -> str | None:
    """The incremental text of one streaming chunk, or None for a chunk that carries no
    content (role-only openers, finish markers)."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    delta = getattr(getattr(choices[0], "delta", None), "content", None)
    return delta if isinstance(delta, str) else None


def _extract_embedding_vectors(response: object) -> list[list[float]]:
    """The embedding vectors from an embeddings response, one per input, in order. An item
    that carries no vector is an error, not a silently-empty result."""
    data = getattr(response, "data", None)
    if not isinstance(data, list) or not data:
        raise LLMError("embedding returned no vectors")
    # The API tags each item with its input `index`; sort by it so the vectors line up with
    # the input order even if the response arrives (or is batched) out of order.
    items = sorted(data, key=lambda item: getattr(item, "index", 0))
    vectors: list[list[float]] = []
    for item in items:
        embedding = getattr(item, "embedding", None)
        if not isinstance(embedding, list) or not embedding:
            raise LLMError("embedding response held an item with no vector")
        vectors.append(list(embedding))
    return vectors
