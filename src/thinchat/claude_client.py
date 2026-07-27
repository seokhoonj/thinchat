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

import thinchat.keys as keys
from thinchat.client import Capability, _BaseClient
from thinchat.errors import LLMError, ProviderUnavailableError

__all__ = ["ClaudeClient"]

_CLAUDE_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

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

    def __init__(self, *, api_key: str, model: str, max_tokens: int = _CLAUDE_DEFAULT_MAX_TOKENS) -> None:
        try:
            import httpx
            from anthropic import Anthropic, AnthropicError
        except ImportError as err:
            raise ProviderUnavailableError(
                "the claude client needs the anthropic package; install it with: "
                "pip install 'thinchat[claude]'"
            ) from err
        self.model          = model
        self.capabilities   = _CAPABILITIES
        self._api_key       = api_key
        self._max_tokens    = max_tokens
        # Catch only Anthropic's own error family (so our bugs surface as themselves); the
        # streaming path also raises raw httpx on a mid-stream transport drop.
        self._sdk_error     = AnthropicError
        self._stream_errors = (AnthropicError, httpx.HTTPError)
        self._client        = Anthropic(api_key=api_key)
        self._aclient       = None   # built on first async use (see _make_aclient)

    def _make_aclient(self) -> Any:
        from anthropic import AsyncAnthropic  # the sync import above already proved it installed
        return AsyncAnthropic(api_key=self._api_key)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        try:
            response = self._client.messages.create(**self._make_request(prompt, system))
        except self._sdk_error as err:
            raise LLMError(f"claude completion failed: {err}") from err
        return _extract_message_text(response)

    async def acomplete(self, prompt: str, *, system: str | None = None) -> str:
        try:
            response = await self._get_aclient().messages.create(**self._make_request(prompt, system))
        except self._sdk_error as err:
            raise LLMError(f"claude completion failed: {err}") from err
        return _extract_message_text(response)

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        try:
            with self._client.messages.stream(**self._make_request(prompt, system)) as events:
                yield from events.text_stream
        except self._stream_errors as err:
            raise LLMError(f"claude stream failed: {err}") from err

    async def astream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        try:
            async with self._get_aclient().messages.stream(**self._make_request(prompt, system)) as events:
                async for chunk in events.text_stream:
                    yield chunk
        except self._stream_errors as err:
            raise LLMError(f"claude stream failed: {err}") from err

    def _make_request(self, prompt: str, system: str | None) -> dict[str, object]:
        request: dict[str, object] = {
            "model":      self.model,
            "max_tokens": self._max_tokens,
            "messages":   [{"role": "user", "content": prompt}],
        }
        if system is not None:
            request["system"] = system   # Anthropic takes system as its own field, not a message
        return request


def _make_claude_client(
    *, model: str | None = None, api_key: str | None = None, max_tokens: int | None = None,
) -> ClaudeClient:
    """Construct the Claude client. ``max_tokens`` caps the reply; when None the default
    (Anthropic requires the field) is used.

    Raises:
        ProviderUnavailableError: the anthropic SDK is not installed, or no API key is set
            (or passed).
    """
    key = api_key if api_key is not None else keys.get_api_key("claude")
    if not key:
        raise ProviderUnavailableError(
            f"no API key for claude: set {keys.ENV_BY_PROVIDER['claude']} or pass api_key="
        )
    return ClaudeClient(
        api_key    = key,
        model      = model or _CLAUDE_DEFAULT_MODEL,
        max_tokens = max_tokens if max_tokens is not None else _CLAUDE_DEFAULT_MAX_TOKENS,
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
