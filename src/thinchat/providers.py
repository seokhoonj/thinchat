"""Name a provider, get a client.

The one place that maps each provider name to its client factory, so a caller -- a CLI, a
config value, another package -- asks for a client by provider string and never re-authors
the mapping. The roster (``PROVIDERS``) is *derived* from the factory map, so the two
cannot drift. Adding a provider is one entry here.
"""

from __future__ import annotations

from functools import partial
from typing import Protocol

from thinchat.claude_client import _make_claude_client
from thinchat.client import Client, Provider
from thinchat.errors import UnknownProviderError
from thinchat.openai_client import _make_openai_client

__all__ = ["PROVIDERS", "make_client"]


class _ClientFactory(Protocol):
    """The call shape every provider factory shares once its provider name is bound: the
    override knobs (model, api_key, and the generation/transport settings) in, a client out.
    Typed (not ``Callable[..., Client]``) so a call with a mistyped kwarg is caught
    statically."""

    def __call__(
        self, *, model: str | None = None, api_key: str | None = None,
        max_tokens: int | None = None, temperature: float | None = None,
        top_p: float | None = None, timeout: float | None = None,
        max_retries: int | None = None,
    ) -> Client: ...


# One factory per provider -- the single source of the roster. claude has its own factory;
# openai/gemini/ollama share the OpenAI-compatible one (bound to their name).
_FACTORY_BY_PROVIDER: dict[Provider, _ClientFactory] = {
    "claude": _make_claude_client,
    "openai": partial(_make_openai_client, "openai"),
    "gemini": partial(_make_openai_client, "gemini"),
    "ollama": partial(_make_openai_client, "ollama"),
}

# The providers thinchat supports, in insertion order -- derived from the factory map so
# the roster and the dispatch never drift.
PROVIDERS: tuple[Provider, ...] = tuple(_FACTORY_BY_PROVIDER)


def make_client(
    provider: str, *, model: str | None = None, api_key: str | None = None,
    max_tokens: int | None = None, temperature: float | None = None,
    top_p: float | None = None, timeout: float | None = None,
    max_retries: int | None = None,
) -> Client:
    """Construct the client for ``provider`` (one of ``PROVIDERS``). ``model`` overrides the
    provider's default model; ``api_key`` overrides the environment key (for tests, or a
    caller that manages its own secrets).

    The remaining knobs are the settings every provider exposes under the same name, sent
    only when set (None leaves the provider's own default in place -- except ``max_tokens``,
    which claude requires and so falls back to a default there):

    - ``max_tokens`` caps the reply length (a positive integer).
    - ``temperature`` / ``top_p`` steer sampling; they go in the request.
    - ``timeout`` (seconds) and ``max_retries`` configure the HTTP client -- how long to
      wait for a reply and how many times the SDK retries a transient failure.

    Raises:
        UnknownProviderError: ``provider`` is not one of ``PROVIDERS``.
        ProviderUnavailableError: the provider's SDK is not installed, or it needs a key
            and none is available.
    """
    if provider not in _FACTORY_BY_PROVIDER:
        raise UnknownProviderError(
            f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}"
        )
    factory = _FACTORY_BY_PROVIDER[provider]   # `provider` narrowed to a known key by the check above
    return factory(
        model=model, api_key=api_key, max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, timeout=timeout, max_retries=max_retries,
    )
