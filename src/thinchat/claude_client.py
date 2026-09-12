"""The Claude client, over Anthropic's native SDK.

Claude is the one client not reached through the OpenAI-compatible path: Anthropic's API
takes ``system`` as its own argument rather than a message, and Anthropic has no
first-party embeddings API, so this client has completion, streaming, and structured
output but refuses ``embed`` (the base's default). ``max_tokens`` is required by this API
(unlike OpenAI's), so it is a constructor knob with a generous default.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

from credbox import Secret

import thinchat.keys as keys
from thinchat.client import Capability, _BaseClient, build_sdk_client
from thinchat.errors import LLMError, ProviderUnavailableError
from thinchat.results import Completion, Usage, _as_int

__all__ = ["ClaudeClient"]

_CLAUDE_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# The official Anthropic endpoint, pinned explicitly so the SDK does NOT fall back to reading
# the ANTHROPIC_BASE_URL environment variable. Passing base_url=None would let an attacker who
# controls the env (but cannot read the 0600 key store) set ANTHROPIC_BASE_URL and redirect the
# store-resolved key to their host on the first request. A caller wanting a gateway endpoint
# passes base_url= to make_client explicitly (a code decision, not an ambient env).
_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# No "embeddings": Anthropic has no first-party embeddings API, so embed stays the base's
# refusing default and a caller reads `supports("embeddings")` as False.
_CAPABILITIES: frozenset[Capability] = frozenset({"completion", "streaming", "structured_output"})

# Anthropic requires max_tokens (unlike OpenAI, where it is optional). 4096 is a generous
# default that stays within every current Claude model's output limit; a caller wanting a
# different cap passes max_tokens= to make_client.
_CLAUDE_DEFAULT_MAX_TOKENS = 4096


class ClaudeClient(_BaseClient):
    """A client over Anthropic's ``anthropic`` SDK. Holds a sync client and lazily builds a
    reusable async client on first async use, plus the model id. ``max_tokens`` bounds every
    reply (the API requires it)."""

    _sdk_error:     type[Exception]
    _stream_errors: tuple[type[Exception], ...]
    _secret_key:    Secret   # claude always has a real key (never None, unlike the base's union)

    def __init__(
        self, *, api_key: Secret, model: str, base_url: str,
        max_tokens: int = _CLAUDE_DEFAULT_MAX_TOKENS,
        temperature: float | None = None, top_p: float | None = None,
        timeout: float | None = None, max_retries: int | None = None,
    ) -> None:
        try:
            import httpx
            from anthropic import Anthropic, AnthropicError
            from anthropic import RateLimitError as _AnthropicRateLimitError
        except ImportError as err:
            raise ProviderUnavailableError(
                "the anthropic package is required but could not be imported; reinstall thinchat"
            ) from err
        self.model          = model
        self.capabilities   = _CAPABILITIES
        self._base_url      = base_url   # pinned official endpoint (or an explicit override)
        self._secret_key    = api_key   # a Secret; claude always has a real key, scrubbed from errors
        self._max_tokens    = max_tokens
        self._temperature   = temperature   # None -> omit (sampling knobs go in the request)
        self._top_p         = top_p
        self._timeout       = timeout        # None -> the SDK default (transport, on the client)
        self._max_retries   = max_retries
        # Catch only Anthropic's own error family (so our bugs surface as themselves); the
        # streaming path also raises raw httpx on a mid-stream transport drop.
        self._sdk_error       = AnthropicError
        self._ratelimit_error = _AnthropicRateLimitError
        self._provider_label  = "claude"
        self._stream_errors   = (AnthropicError, httpx.HTTPError)
        # Reveal the Secret only here and in the async builder -- the two SDK-construction points.
        # Build via build_sdk_client so a constructor failure (e.g. a malformed ALL_PROXY that
        # httpx rejects) never escapes with the revealed key live in the SDK's __init__ frame.
        self._client          = build_sdk_client(
            lambda: Anthropic(base_url=base_url, api_key=self._secret_key.reveal(), **self._transport_kwargs()),
            provider="claude",
        )
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
        from anthropic import AsyncAnthropic  # the sync import above already proved it installed
        return build_sdk_client(
            lambda: AsyncAnthropic(base_url=self._base_url, api_key=self._secret_key.reveal(),
                                   **self._transport_kwargs()),
            provider="claude",
        )

    def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        try:
            response = self._client.messages.create(**self._make_request(prompt, system))
        except self._sdk_error as err:
            failure = self._map_sdk_failure(err, "completion")
        else:
            return _completion_from_message(response)
        raise failure   # outside the except: the SDK error (key in its request headers/frame) is not chained

    async def acomplete(self, prompt: str, *, system: str | None = None) -> Completion:
        try:
            response = await self._get_aclient().messages.create(**self._make_request(prompt, system))
        except self._sdk_error as err:
            failure = self._map_sdk_failure(err, "completion")
        else:
            return _completion_from_message(response)
        raise failure

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        failure = None
        try:
            with self._client.messages.stream(**self._make_request(prompt, system)) as events:
                yield from events.text_stream
        except self._stream_errors as err:
            failure = self._map_sdk_failure(err, "stream")
        if failure is not None:   # raise outside the except: no SDK error chained (its headers hold the key)
            raise failure

    async def astream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        failure = None
        try:
            async with self._get_aclient().messages.stream(**self._make_request(prompt, system)) as events:
                async for chunk in events.text_stream:
                    yield chunk
        except self._stream_errors as err:
            failure = self._map_sdk_failure(err, "stream")
        if failure is not None:
            raise failure

    def _make_request(self, prompt: str, system: str | None) -> dict[str, object]:
        request: dict[str, object] = {
            "model":      self.model,
            "max_tokens": self._max_tokens,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature
        if self._top_p is not None:
            request["top_p"] = self._top_p
        if system is not None:
            request["system"] = system   # Anthropic takes system as its own field, not a message
        return request


def _make_claude_client(
    *, model: str | None = None, api_key: str | None = None, base_url: str | None = None,
    max_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None,
    timeout: float | None = None, max_retries: int | None = None,
) -> ClaudeClient:
    """Construct the Claude client. ``max_tokens`` caps the reply; when None the default
    (Anthropic requires the field) is used. The other settings are sent only when set.

    ``base_url`` (when given) points the client at a gateway/proxy endpoint. When None, the
    official Anthropic endpoint is pinned so the SDK does NOT read ``ANTHROPIC_BASE_URL`` from
    the environment -- an attacker with env-write but no read of the 0600 key store could
    otherwise redirect the resolved key to their host.

    Raises:
        ProviderUnavailableError: the anthropic SDK is not installed, or no API key is set
            (or passed).
        CredentialStoreError: the stored-key file is present but unreadable or malformed
            (propagated from the key store).
    """
    key = keys.get_api_key("claude", override=api_key)
    if key is None:
        raise ProviderUnavailableError(
            f"no API key for claude: pass api_key=, set {keys.ENV_BY_PROVIDER['claude']}, "
            f"or run 'thinchat set claude'"
        )
    return ClaudeClient(
        api_key     = key,
        model       = model or _CLAUDE_DEFAULT_MODEL,
        base_url    = base_url or _ANTHROPIC_BASE_URL,   # never None -> the SDK never reads its env var
        max_tokens  = max_tokens if max_tokens is not None else _CLAUDE_DEFAULT_MAX_TOKENS,
        temperature = temperature,
        top_p       = top_p,
        timeout     = timeout,
        max_retries = max_retries,
    )


def _extract_message_text(response: object) -> str:
    """The concatenated text blocks of a Messages response. Raises ``LLMError`` when the
    reply is empty, or carries content but no text block (e.g. a non-text block only)."""
    content = getattr(response, "content", None)
    if not isinstance(content, list) or not content:
        raise LLMError("claude completion returned an empty reply")
    parts = [
        block.text for block in content
        if getattr(block, "type", None) == "text" and isinstance(getattr(block, "text", None), str)
    ]
    text = "".join(parts).strip()
    if not text:
        raise LLMError("claude reply carried no text block")
    return text


def _completion_from_message(response: object) -> Completion:
    """Build a ``Completion`` from a Messages response: the validated text plus the stop reason,
    token usage, and model where present (all read defensively)."""
    text = _extract_message_text(response)
    stop = getattr(response, "stop_reason", None)
    usage_obj = getattr(response, "usage", None)
    usage = Usage(
        input_tokens=_as_int(getattr(usage_obj, "input_tokens", None)),
        output_tokens=_as_int(getattr(usage_obj, "output_tokens", None)),
    ) if usage_obj is not None else None
    model = getattr(response, "model", None)
    return Completion(
        text,
        finish_reason=stop if isinstance(stop, str) else None,
        truncated=(stop == "max_tokens"),   # Anthropic signals a token-cap cutoff with "max_tokens"
        usage=usage,
        model=model if isinstance(model, str) else None,
    )
