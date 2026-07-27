"""Resolve a client's API key from the environment.

thinchat is a library, not an application, so it reads keys from the environment (or an
explicit ``api_key=`` a caller passes) and never imposes a file location of its own --
where keys live on disk is the calling program's decision. Each name follows the provider
handle you pass to ``make_client`` -- ``<PROVIDER>_API_KEY`` -- so ``make_client("claude")``
pairs with ``CLAUDE_API_KEY``, matching the roster rather than each vendor SDK's own default.
"""

from __future__ import annotations

import os

__all__ = ["ENV_BY_PROVIDER", "get_api_key"]

# The environment variable each provider's key is read from. ollama is absent on purpose:
# a local server needs no key, so its client passes a dummy the SDK accepts.
ENV_BY_PROVIDER: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "claude": "CLAUDE_API_KEY",
}


def get_api_key(provider: str) -> str | None:
    """Return the API key for ``provider`` from its environment variable, or ``None`` when
    it is unset (so a client can phrase its own "no key" error) or the provider needs
    none (ollama). A ``provider`` outside the roster simply reads as "no env"."""
    name = ENV_BY_PROVIDER.get(provider)
    return os.environ.get(name) if name is not None else None
