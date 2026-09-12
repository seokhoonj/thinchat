"""Domain exception hierarchy for thinchat.

Every error thinchat raises on purpose derives from ``ThinchatError``, so a caller can
handle this package's failures with one ``except`` without catching unrelated bugs. The
tree separates the ways a call can fail: the client could not be built
(``ProviderUnavailableError``), the name is not one we know (``UnknownProviderError``), the
operation is not available for the provider (``UnsupportedError`` -- a missing capability, or
a key for a keyless provider), a key to store was blank (``BlankKeyError``), the stored-key
store could not be read or written (``CredentialStoreError``), or the API call itself failed
(``LLMError``). One refinement of that last case is broken out: a rate limit
(HTTP 429) that outlives the SDK's own retries surfaces as ``RateLimitError``, an
``LLMError`` subclass carrying the requested retry delay -- so ``except LLMError`` still
catches it, while a caller that wants to wait and retry can catch it by its own type.
"""

from __future__ import annotations

__all__ = [
    "ThinchatError",
    "ProviderUnavailableError",
    "LLMError",
    "AuthError",
    "RateLimitError",
    "UnknownProviderError",
    "UnsupportedError",
    "BlankKeyError",
    "CredentialStoreError",
]


class ThinchatError(Exception):
    """Base for every error thinchat raises deliberately."""


class UnknownProviderError(ThinchatError):
    """The requested client name is not one of the four thinchat supports. Permanent --
    a typo or an unsupported provider -- so the caller fixes the name rather than retries."""


class ProviderUnavailableError(ThinchatError):
    """A client could not be constructed: its SDK is not installed (reinstall thinchat -- both
    provider SDKs ship with it) or no API key is available for it. Distinct from ``LLMError``
    because nothing was sent -- the failure is local setup, not the remote service."""


class UnsupportedError(ThinchatError):
    """A requested operation is not available for this provider. Two cases: a capability the
    client lacks -- asking Claude for embeddings, which Anthropic has no first-party API for,
    so ``supports`` lets a caller check before calling -- or storing/removing a key for a
    provider that needs none (ollama). Permanent either way, so the caller changes the request
    rather than retrying."""


class BlankKeyError(ThinchatError, ValueError):
    """A key handed to ``set_api_key`` was empty or whitespace. A blank key would store as
    present yet resolve as absent (the store treats blank as unset), an inconsistency the
    store rejects. Also a ``ValueError`` -- the natural type for a bad argument value -- so a
    caller guarding with ``except ValueError`` still catches it, while ``except ThinchatError``
    keeps it inside the package's hierarchy."""


class CredentialStoreError(ThinchatError):
    """thinchat's stored-key file could not be read or written -- present but unreadable or
    malformed on a read, or the write failed. Wraps the underlying storage error so a caller
    handles it as a thinchat failure (``except ThinchatError``) without importing the storage
    backend's own exception type. Distinct from ``ProviderUnavailableError`` (no key found is
    a normal, empty result -- this is the store itself being broken)."""


class LLMError(ThinchatError):
    """The API call was made but failed: the service returned an error, or the reply was
    empty or not the requested shape. Its message carries the underlying cause. ``status_code``
    is the HTTP status the service returned when one was available (None for a transport
    failure, or an empty / malformed reply) -- a caller can branch on it to tell a transient
    5xx worth retrying from a permanent 4xx, without matching on the message text."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(LLMError):
    """The service rejected the credentials (HTTP 401/403): the API key is missing, invalid,
    revoked, or lacks access to the resource. Permanent -- fix the key rather than retry. A
    subclass of ``LLMError`` so ``except LLMError`` still catches it; catch this type to tell a
    bad key from a transient failure."""


class RateLimitError(LLMError):
    """The service refused the call for rate limiting (HTTP 429) after its own retries were
    exhausted. ``retry_after`` is the requested wait in seconds when the response's
    ``Retry-After`` header was a plain number, otherwise None. A subclass of ``LLMError`` so
    existing handlers still catch it; catch this type to distinguish a transient limit worth
    waiting and retrying from a permanent failure."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after
