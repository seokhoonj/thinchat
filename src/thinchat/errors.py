"""Domain exception hierarchy for thinchat.

Every error thinchat raises on purpose derives from ``ThinchatError``, so a caller can
handle this package's failures with one ``except`` without catching unrelated bugs. The
tree separates the four ways a call can fail: the client could not be built
(``ProviderUnavailableError``), the name is not one we know (``UnknownProviderError``), the
client does not offer the requested capability (``UnsupportedError``), or the API call
itself failed (``LLMError``).
"""

from __future__ import annotations

__all__ = [
    "ThinchatError",
    "ProviderUnavailableError",
    "LLMError",
    "UnknownProviderError",
    "UnsupportedError",
]


class ThinchatError(Exception):
    """Base for every error thinchat raises deliberately."""


class UnknownProviderError(ThinchatError):
    """The requested client name is not one of the four thinchat supports. Permanent --
    a typo or an unsupported provider -- so the caller fixes the name rather than retries."""


class ProviderUnavailableError(ThinchatError):
    """A client could not be constructed: its SDK is not installed (install the matching
    extra) or no API key is available for it. Distinct from ``LLMError`` because nothing
    was sent -- the failure is local setup, not the remote service."""


class UnsupportedError(ThinchatError):
    """The client does not offer the requested capability -- asking Claude for embeddings,
    which Anthropic has no first-party API for. Permanent for that client, so ``supports``
    lets a caller check before calling."""


class LLMError(ThinchatError):
    """The API call was made but failed: the service returned an error, or the reply was
    empty or not the requested shape. Its message carries the underlying cause."""
