"""Resolve, store, and enumerate a provider's API key.

thinchat is a library first: a caller that passes ``api_key=`` to ``make_client`` never
touches a file, and the ``<PROVIDER>_API_KEY`` environment variable still works with nothing
configured. On top of that, thinchat can persist a key to its own store
(``~/.config/thinchat/credentials.json``, mode 0600) so a user saves it once with
``thinchat set <provider>`` instead of exporting it every session. The storage, the
permission hardening, and the env-over-file resolution are delegated to xdg-kit; this module
only maps thinchat's provider handles onto that store.

The store is keyed by the same ``<PROVIDER>_API_KEY`` name the environment uses, so a key set
in either place resolves identically, and a sibling package that writes thinchat's store (a
``newswatcher setup`` wizard calling ``set_api_key``) shares one source for the LLM key
rather than keeping its own copy.
"""

from __future__ import annotations

from typing import get_args

from xdg_kit import XdgKitError, get_secret, secret_names, set_secret, unset_secret

from thinchat.client import Provider
from thinchat.errors import CredentialStoreError, UnknownProviderError, UnsupportedError

__all__ = [
    "ENV_BY_PROVIDER",
    "get_api_key",
    "set_api_key",
    "unset_api_key",
    "stored_providers",
    "stored_key_name",
]

# The xdg-kit app whose store thinchat's keys live in: ~/.config/thinchat/credentials.json.
APP = "thinchat"

# The environment variable each provider's key is read from, and the name it is stored under.
# ollama is absent on purpose: a local server needs no key, so its client passes a dummy the
# SDK accepts. Consumers import this map by name -- keep it here.
ENV_BY_PROVIDER: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "claude": "CLAUDE_API_KEY",
}

# Every provider make_client knows, so the read path can reject a typo while still resolving a
# known-but-keyless provider (ollama). Derived from the one Provider definition so the two
# never drift; the ordered tuple gives messages a stable order (frozenset iteration has none).
_ALL_PROVIDERS_ORDER: tuple[str, ...] = get_args(Provider)
_ALL_PROVIDERS: frozenset[str] = frozenset(_ALL_PROVIDERS_ORDER)


def get_api_key(provider: str, *, override: str | None = None) -> str | None:
    """Resolve ``provider``'s API key across ``override`` > env > stored file, or ``None``
    when it is unset everywhere (so a client can phrase its own "no key" error) or the
    provider needs none (ollama). ``override`` is ``make_client``'s ``api_key=``, kept as the
    top tier so a caller managing its own secrets never reads the store.

    Raises:
        UnknownProviderError: ``provider`` is not one thinchat supports (a typo resolves to
            an error, not a misleading "no key").
        CredentialStoreError: the store could not be read -- present but unreadable or
            malformed, or the storage backend failed (propagated from xdg-kit).
    """
    if provider not in _ALL_PROVIDERS:
        raise UnknownProviderError(
            f"unknown provider {provider!r}; choose one of {', '.join(_ALL_PROVIDERS_ORDER)}")
    name = ENV_BY_PROVIDER.get(provider)
    if name is None:                      # ollama: known, but needs no key
        return override
    try:
        return get_secret(APP, name, override=override)
    except XdgKitError as err:
        raise CredentialStoreError(f"could not read the stored key for {provider}") from err


def set_api_key(provider: str, *, value: str) -> None:
    """Store ``value`` as ``provider``'s API key in thinchat's own store (mode 0600).
    ``value`` is keyword-only so it cannot be swapped with ``provider``.

    Raises:
        UnknownProviderError: ``provider`` is not one thinchat supports.
        UnsupportedError: ``provider`` needs no API key (ollama), so none can be stored.
        CredentialStoreError: the store could not be written.
    """
    name = stored_key_name(provider)
    try:
        set_secret(APP, name, value=value)
    except XdgKitError as err:
        raise CredentialStoreError(f"could not store the key for {provider}") from err


def unset_api_key(provider: str) -> None:
    """Remove ``provider``'s stored key; a no-op when none is stored. Only thinchat's own
    store is touched -- never the environment.

    Raises:
        UnknownProviderError: ``provider`` is not one thinchat supports.
        UnsupportedError: ``provider`` needs no API key (ollama).
        CredentialStoreError: the store could not be written.
    """
    name = stored_key_name(provider)
    try:
        unset_secret(APP, name)
    except XdgKitError as err:
        raise CredentialStoreError(f"could not remove the stored key for {provider}") from err


def stored_providers() -> list[str]:
    """The providers that have a key in thinchat's store, in ``ENV_BY_PROVIDER`` order --
    never the values. A provider whose key is only in the environment is not listed: this
    reports the file store alone, so a caller (a setup wizard, the CLI's ``list``) can show
    what has been saved.

    Raises:
        CredentialStoreError: the store is present but malformed.
    """
    try:
        stored = set(secret_names(APP))
    except XdgKitError as err:
        raise CredentialStoreError("could not read the credential store") from err
    return [provider for provider, name in ENV_BY_PROVIDER.items() if name in stored]


def stored_key_name(provider: str) -> str:
    """The name a key for ``provider`` is filed under in the store and the environment, or
    raise if ``provider`` takes no stored key. Shared by the write paths (``set``/``unset``)
    and by a caller that wants to validate a provider *before* prompting for a secret (the
    CLI ``set``), so a typo costs no wasted entry. ollama is known but keyless
    (``UnsupportedError``); anything off the roster is a typo (``UnknownProviderError``).

    Raises:
        UnknownProviderError: ``provider`` is not one thinchat supports.
        UnsupportedError: ``provider`` is known but needs no API key (ollama).
    """
    name = ENV_BY_PROVIDER.get(provider)
    if name is not None:
        return name
    if provider in _ALL_PROVIDERS:
        raise UnsupportedError(f"{provider} needs no API key, so none can be stored")
    raise UnknownProviderError(
        f"unknown provider {provider!r}; choose one of {', '.join(ENV_BY_PROVIDER)}")
