"""Name a provider, get a client.

The one place that maps each provider name to its client factory, so a caller -- a CLI, a
config value, another package -- asks for a client by provider string and never re-authors
the mapping. The roster (``PROVIDERS``) is *derived* from the factory map, so the two
cannot drift. Adding a provider is one entry here.
"""

from __future__ import annotations

from functools import partial
from typing import Protocol, get_args

from thinchat.claude_client import _make_claude_client
from thinchat.client import Client, Provider
from thinchat.errors import UnknownProviderError
from thinchat.openai_client import _SPEC_BY_PROVIDER, _make_openai_client

__all__ = ["PROVIDERS", "make_client"]


class _ClientFactory(Protocol):
    """The call shape every provider factory shares once its provider name is bound: the
    override knobs (model, api_key, and the generation/transport settings) in, a client out.
    Typed (not ``Callable[..., Client]``) so a call with a mistyped kwarg is caught
    statically."""

    def __call__(
        self, *, model: str | None = None, api_key: str | None = None,
        base_url: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None, top_p: float | None = None,
        timeout: float | None = None, max_retries: int | None = None,
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

# Import-time coherence (a raise, not assert -- survives `python -O`): the Provider type, the
# factory roster, and the OpenAI-compat spec table are separate registries. Tie them so a
# provider added to one but not the others fails here at import, not at a user's make_client()
# call (a provider in the type without a factory would be typed-but-unusable; an OpenAI-compat
# factory without a spec would be advertised in PROVIDERS but raise at construction).
if frozenset(_FACTORY_BY_PROVIDER) != frozenset(get_args(Provider)):
    raise RuntimeError("the Provider type and the make_client factory roster disagree")
_compat_bound = frozenset(
    factory.args[0]
    for factory in _FACTORY_BY_PROVIDER.values()
    if isinstance(factory, partial) and factory.func is _make_openai_client
)
if _compat_bound != frozenset(_SPEC_BY_PROVIDER):
    raise RuntimeError("the OpenAI-compatible factory roster and _SPEC_BY_PROVIDER disagree")


def make_client(
    provider: str, *, model: str | None = None, api_key: str | None = None,
    base_url: str | None = None, max_tokens: int | None = None,
    temperature: float | None = None, top_p: float | None = None,
    timeout: float | None = None, max_retries: int | None = None,
) -> Client:
    """Construct the client for ``provider`` (one of ``PROVIDERS``). ``model`` overrides the
    provider's default model; ``api_key`` overrides the environment key (for tests, or a
    caller that manages its own secrets).

    The remaining knobs are the settings every provider exposes under the same name, sent
    only when set (None leaves the provider's own default in place -- except ``max_tokens``,
    which claude requires and so falls back to a default there):

    - ``base_url`` points the client at a gateway/proxy/Azure endpoint. When None, each
      provider pins its official endpoint, so the vendor SDK never reads its ``*_BASE_URL``
      environment variable -- an attacker who can only write the environment (not read the
      0600 key store) cannot redirect the resolved key to their host. (ollama is keyless, so
      it has no key to leak; when ``base_url`` is None it defaults to ``OLLAMA_HOST``.)
    - ``max_tokens`` caps the reply length (a positive integer).
    - ``temperature`` / ``top_p`` steer sampling; they go in the request.
    - ``timeout`` (seconds) and ``max_retries`` configure the HTTP client -- how long to
      wait for a reply and how many times the SDK retries a transient failure.

    Raises:
        UnknownProviderError: ``provider`` is not one of ``PROVIDERS``.
        ProviderUnavailableError: the provider's SDK is not installed, or it needs a key
            and none is available.
        CredentialStoreError: the stored-key file is present but unreadable or malformed
            (propagated from the key store).
    """
    if provider not in _FACTORY_BY_PROVIDER:
        raise UnknownProviderError(
            f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}"
        )
    factory = _FACTORY_BY_PROVIDER[provider]   # `provider` narrowed to a known key by the check above
    return factory(
        model=model, api_key=api_key, base_url=base_url, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, timeout=timeout, max_retries=max_retries,
    )
