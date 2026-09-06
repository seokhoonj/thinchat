"""thinchat: a thin, unified client for four LLM providers -- claude, openai, gemini, ollama.

Name a provider, then call it. Each client offers completion (whole, streamed, or
JSON-structured) and, where the provider has one, embeddings -- with an async twin for each.

    from thinchat import make_client

    llm = make_client("claude")                 # key from CLAUDE_API_KEY
    print(llm.complete("Say hi in one word."))
    verdict = llm.parse("Is this an ad? 'Buy now, 50% off'",
                        schema={"type": "object",
                                "properties": {"is_ad": {"type": "boolean"}},
                                "required": ["is_ad"]})

Installing thinchat brings both provider SDKs (openai and anthropic), so every provider
works out of the box; each SDK is imported lazily when you first construct its client.
"""

from importlib.metadata import PackageNotFoundError, version

from thinchat.claude_client import ClaudeClient
from thinchat.client import Capability, Client, Provider
from thinchat.errors import (
    CredentialStoreError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
    ThinchatError,
    UnknownProviderError,
    UnsupportedError,
)
from thinchat.keys import get_api_key, set_api_key, stored_providers, unset_api_key
from thinchat.openai_client import OpenAICompatibleClient
from thinchat.providers import PROVIDERS, make_client

__all__ = [
    "PROVIDERS",
    "Capability",
    "ClaudeClient",
    "Client",
    "CredentialStoreError",
    "LLMError",
    "RateLimitError",
    "OpenAICompatibleClient",
    "Provider",
    "ProviderUnavailableError",
    "ThinchatError",
    "UnknownProviderError",
    "UnsupportedError",
    "get_api_key",
    "make_client",
    "set_api_key",
    "stored_providers",
    "unset_api_key",
]

try:
    __version__ = version("thinchat")   # single source of truth: the installed metadata
except PackageNotFoundError:            # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
