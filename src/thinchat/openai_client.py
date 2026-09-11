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

from credbox import Secret

import thinchat.keys as keys
from thinchat.client import Capability, _BaseClient, build_sdk_client
from thinchat.errors import LLMError, ProviderUnavailableError, UnknownProviderError

__all__ = ["OpenAICompatibleClient"]

# ollama runs locally and needs no key, but the OpenAI SDK requires a non-empty one, so its
# client is built with this placeholder. It is not a secret, so it is never a key to scrub.
_OLLAMA_DUMMY_KEY = "ollama"

# The official OpenAI endpoint, pinned explicitly so the SDK does NOT fall back to reading
# the OPENAI_BASE_URL environment variable. Passing base_url=None would let an attacker who
# controls the env (but cannot read the 0600 key store) set OPENAI_BASE_URL and redirect the
# store-resolved key to their host on the first request. A caller wanting a gateway/Azure
# endpoint passes base_url= to make_client explicitly (a code decision, not an ambient env).
_OPENAI_BASE_URL = "https://api.openai.com/v1"

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
    ``max_tokens_field`` is the request field the reply cap is sent under: OpenAI's newer
    (o-series / GPT-5-class) models reject ``max_tokens`` and require ``max_completion_tokens``,
    while the Gemini and Ollama compat layers take the original ``max_tokens`` -- so a provider
    quirk stays data here rather than an ``if provider ==`` branch in the request builder.
    Keyword-only so two same-typed model ids cannot be transposed by position."""

    base_url:        str | None
    chat_model:      str
    embed_model:     str
    has_native_json: bool
    needs_key:       bool
    is_local:        bool
    max_tokens_field: str


# The three providers, as data. A `-latest` alias for Gemini, not a pinned name: Google
# gates pinned model ids (e.g. gemini-2.5-flash-lite) to 404 for newly created API keys,
# while the alias tracks the current model and stays on the free tier. Private: reached
# only through _make_openai_client, not exported.
_SPEC_BY_PROVIDER: dict[str, _ProviderSpec] = {
    "openai": _ProviderSpec(
        base_url=None, chat_model="gpt-4o-mini", embed_model="text-embedding-3-small",
        has_native_json=True, needs_key=True, is_local=False,
        max_tokens_field="max_completion_tokens"),
    "gemini": _ProviderSpec(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        chat_model="gemini-flash-lite-latest", embed_model="gemini-embedding-001",
        has_native_json=True, needs_key=True, is_local=False,
        max_tokens_field="max_tokens"),
    "ollama": _ProviderSpec(
        base_url=None, chat_model="llama3.1", embed_model="nomic-embed-text",
        has_native_json=False, needs_key=False, is_local=True,
        max_tokens_field="max_tokens"),
}

# Coherence: each spec's needs_key must match keys.py's keyed/keyless classification. Otherwise
# the missing-key path below indexes keys.ENV_BY_PROVIDER[provider] and would raise a raw
# KeyError inside the error message (needs_key True but keyless there), or skip the key check
# entirely (needs_key False but keyed there). Enforced at import with a raise (survives -O).
for _provider, _spec in _SPEC_BY_PROVIDER.items():
    if _spec.needs_key != (_provider in keys.ENV_BY_PROVIDER):
        raise RuntimeError(
            f"provider {_provider!r}: _ProviderSpec.needs_key disagrees with keys.ENV_BY_PROVIDER")


class OpenAICompatibleClient(_BaseClient):
    """A client over the ``openai`` SDK, pointed at any OpenAI-compatible endpoint. Holds a
    sync client and lazily builds a reusable async client on first async use, plus the chat
    and embedding model ids, so the same class serves OpenAI, Gemini, and Ollama."""

    _sdk_error:     type[Exception]
    _stream_errors: tuple[type[Exception], ...]

    def __init__(
        self, *, provider: str, base_url: str | None, api_key: Secret | None, model: str,
        embed_model: str, has_native_json: bool, max_tokens_field: str = "max_tokens",
        max_tokens: int | None = None,
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
        self._secret_key      = api_key   # a Secret, or None for a keyless provider (ollama)
        self._embed_model     = embed_model
        self._has_native_json = has_native_json
        self._max_tokens_field = max_tokens_field   # "max_tokens" or openai's "max_completion_tokens"
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
        # Build via build_sdk_client so a constructor failure (e.g. a malformed ALL_PROXY that
        # httpx rejects) never escapes with the revealed key live in the SDK's __init__ frame.
        self._client          = build_sdk_client(
            lambda: OpenAI(base_url=base_url, api_key=self._sdk_key(), **self._transport_kwargs()),
            provider=provider,
        )
        self._aclient         = None   # built on first async use (see _make_aclient)

    def _sdk_key(self) -> str:
        """The plaintext key the SDK constructor needs -- revealed HERE and nowhere else, so the
        secret lives as a ``Secret`` everywhere else. When there is no key (``_secret_key`` None --
        ollama with no override) the SDK still requires a non-empty string, so it gets the
        non-secret placeholder; a non-blank ollama override, if one was passed, is forwarded as-is."""
        return self._secret_key.reveal() if self._secret_key is not None else _OLLAMA_DUMMY_KEY

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
        return build_sdk_client(
            lambda: AsyncOpenAI(base_url=self._base_url, api_key=self._sdk_key(), **self._transport_kwargs()),
            provider=self._provider,
        )

    def __repr__(self) -> str:   # one class serves three providers; show which
        return f"OpenAICompatibleClient(provider={self._provider!r}, model={self.model!r})"

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        return self._chat(self._make_messages(prompt, system))

    async def acomplete(self, prompt: str, *, system: str | None = None) -> str:
        return await self._achat(self._make_messages(prompt, system))

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        request = self._make_request(self._make_messages(prompt, system), json_mode=False)
        request["stream"] = True
        failure = None
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
            failure = self._map_sdk_failure(err, "stream")
        if failure is not None:   # raise outside the except: no SDK error chained (its headers hold the key)
            raise failure

    async def astream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        request = self._make_request(self._make_messages(prompt, system), json_mode=False)
        request["stream"] = True
        failure = None
        try:
            sdk_stream: Any = await self._get_aclient().chat.completions.create(**request)
            async with sdk_stream as events:
                async for chunk in events:
                    delta = _extract_stream_text(chunk)
                    if delta:
                        yield delta
        except self._stream_errors as err:
            failure = self._map_sdk_failure(err, "stream")
        if failure is not None:
            raise failure

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Return one embedding vector per input text (empty input -> empty list, no call).
        Raises ``LLMError`` if the API call fails or a reply carries no vector."""
        text_list = _as_text_list(texts)
        if not text_list:
            return []   # one vector per input; zero inputs -> zero vectors, not an error
        try:
            response = self._client.embeddings.create(model=model or self._embed_model, input=text_list)
        except self._sdk_error as err:
            failure = self._map_sdk_failure(err, "embedding")
        else:
            return _extract_embedding_vectors(response, expected=len(text_list))
        raise failure

    async def aembed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Async twin of ``embed``."""
        text_list = _as_text_list(texts)
        if not text_list:
            return []
        try:
            response = await self._get_aclient().embeddings.create(model=model or self._embed_model, input=text_list)
        except self._sdk_error as err:
            failure = self._map_sdk_failure(err, "embedding")
        else:
            return _extract_embedding_vectors(response, expected=len(text_list))
        raise failure

    # Native JSON mode where the endpoint honours it, else the base's prompt-steered path.
    def _text_for_parse(self, prompt: str, system: str) -> str:
        return self._chat(self._make_messages(prompt, system), json_mode=self._has_native_json)

    async def _atext_for_parse(self, prompt: str, system: str) -> str:
        return await self._achat(self._make_messages(prompt, system), json_mode=self._has_native_json)

    def _chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            response = self._client.chat.completions.create(**self._make_request(messages, json_mode=json_mode))
        except self._sdk_error as err:
            failure = self._map_sdk_failure(err, "completion")
        else:
            return _extract_chat_text(response)
        raise failure   # outside the except: the SDK error (key in its request headers/frame) is not chained

    async def _achat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        try:
            response = await self._get_aclient().chat.completions.create(**self._make_request(messages, json_mode=json_mode))
        except self._sdk_error as err:
            failure = self._map_sdk_failure(err, "completion")
        else:
            return _extract_chat_text(response)
        raise failure

    def _make_request(self, messages: list[dict[str, str]], *, json_mode: bool) -> dict[str, object]:
        request: dict[str, object] = {"model": self.model, "messages": messages}
        if self._max_tokens is not None:   # optional here, so send it only when set
            # The field name is per-provider data (_ProviderSpec.max_tokens_field): OpenAI's
            # newer models require max_completion_tokens; the compat layers take max_tokens.
            request[self._max_tokens_field] = self._max_tokens
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
    base_url: str | None = None, max_tokens: int | None = None,
    temperature: float | None = None, top_p: float | None = None,
    timeout: float | None = None, max_retries: int | None = None,
) -> OpenAICompatibleClient:
    """Construct one of the OpenAI-compatible clients (openai / gemini / ollama) by name.
    Each setting is sent only when set; when None the provider's own default stands.

    ``base_url`` (when given) points the client at a gateway/proxy/Azure endpoint. When None,
    a fixed official endpoint is pinned per provider so the SDK does NOT read ``OPENAI_BASE_URL``
    from the environment -- an attacker with env-write but no read of the 0600 key store could
    otherwise redirect the resolved key to their host.

    Raises:
        UnknownProviderError: ``provider`` is not an OpenAI-compatible provider.
        ProviderUnavailableError: the openai SDK is not installed, or the provider needs a
            key and none is set (or passed).
        CredentialStoreError: the stored-key file is present but unreadable or malformed
            (propagated from the key store).
    """
    spec = _SPEC_BY_PROVIDER.get(provider)
    if spec is None:
        raise UnknownProviderError(
            f"{provider!r} is not an OpenAI-compatible provider; "
            f"choose one of {', '.join(_SPEC_BY_PROVIDER)}"
        )
    key = keys.get_api_key(provider, override=api_key)
    if spec.needs_key and key is None:
        raise ProviderUnavailableError(
            f"no API key for {provider}: pass api_key=, set {keys.ENV_BY_PROVIDER[provider]}, "
            f"or run 'thinchat set {provider}'"
        )
    # Explicit override wins; else a fixed endpoint (ollama from OLLAMA_HOST; every other
    # provider a pinned official URL) so base_url is never None -> the SDK never reads its
    # own *_BASE_URL env var.
    effective_base_url = base_url or (
        _ollama_base_url() if spec.is_local else (spec.base_url or _OPENAI_BASE_URL)
    )
    return OpenAICompatibleClient(
        provider        = provider,
        base_url        = effective_base_url,
        api_key         = key,   # a Secret, or None for ollama; the client reveals it only for the SDK
        model            = model or spec.chat_model,
        embed_model      = spec.embed_model,
        has_native_json  = spec.has_native_json,
        max_tokens_field = spec.max_tokens_field,
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


def _as_text_list(texts: Sequence[str]) -> list[str]:
    """The inputs as a list, rejecting a bare ``str``. A ``str`` *is* a ``Sequence[str]`` (of its
    own characters), so ``embed("hello")`` would silently request one embedding per character and
    bill for five vectors nobody wanted. A wrong-typed argument is a caller bug, so this is a
    ``TypeError`` -- not a ThinchatError the caller catches as an operational failure."""
    if isinstance(texts, str):
        raise TypeError("embed expects a sequence of strings, not a single string; pass [text]")
    return list(texts)


def _extract_embedding_vectors(response: object, *, expected: int) -> list[list[float]]:
    """The embedding vectors from an embeddings response, one per input, in input order.
    ``expected`` is how many inputs were sent; the response must carry exactly that many
    vectors, one per input. An item with no vector, a wrong count, or an index set that is
    not a one-to-one map onto the inputs is an error -- never a silently misaligned result.

    The API tags each item with its input ``index``; when present, the indices must be a
    permutation of ``range(expected)`` and the vectors are returned sorted by it, so a
    response batched or reordered by the server still lines up with the inputs. A duplicate,
    missing, or out-of-range index would silently pair a vector with the wrong input, so it
    is rejected. An endpoint that omits ``index`` entirely falls back to positional order,
    with the count check still guarding against a short or padded response."""
    data = getattr(response, "data", None)
    if not isinstance(data, list) or not data:
        raise LLMError("embedding returned no vectors")
    if len(data) != expected:
        raise LLMError(
            f"embedding response returned {len(data)} vectors for {expected} inputs")
    indices = [getattr(item, "index", None) for item in data]
    present = [index for index in indices if index is not None]
    if present:
        # The endpoint tagged items with indices: every item must carry one, and together they
        # must be an exact permutation of range(expected). Requiring the full count first also
        # rules out a mixed response (some items indexed, some not) whose order is ambiguous.
        # sorted(present) == range(expected) then holds only for a true permutation, so a
        # duplicate, missing, or out-of-range index is rejected here.
        if len(present) != expected or sorted(present) != list(range(expected)):
            raise LLMError("embedding response indices did not map one-to-one to the inputs")
        by_index = {index: item for index, item in zip(indices, data, strict=True) if index is not None}
        items = [by_index[position] for position in range(expected)]
    else:
        items = data   # no indices supplied: trust positional order (count already checked)
    vectors: list[list[float]] = []
    for item in items:
        embedding = getattr(item, "embedding", None)
        if not isinstance(embedding, list) or not embedding:
            raise LLMError("embedding response held an item with no vector")
        vectors.append(list(embedding))
    return vectors
