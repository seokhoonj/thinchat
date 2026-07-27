"""thinchat: a tiny, unified client for four LLM providers -- openai, claude, gemini, ollama.

Name a provider, then call it. Each client offers completion (whole, streamed, or
JSON-structured) and, where the provider has one, embeddings -- with an async twin for each.

    from thinchat import make_client

    llm = make_client("gemini")                 # key from GEMINI_API_KEY
    print(llm.complete("Say hi in one word."))
    verdict = llm.parse("Is this an ad? 'Buy now, 50% off'",
                        schema={"type": "object",
                                "properties": {"is_ad": {"type": "boolean"}},
                                "required": ["is_ad"]})

The base install carries no provider SDK: install the client you use -- ``thinchat[openai]``
(covers openai, gemini, and ollama) or ``thinchat[claude]`` -- and it is imported lazily
when you first construct that client.
"""

from importlib.metadata import PackageNotFoundError, version

from thinchat.claude_client import ClaudeClient
from thinchat.client import Capability, Client, Provider
from thinchat.errors import (
    LLMError,
    ProviderUnavailableError,
    ThinchatError,
    UnknownProviderError,
    UnsupportedError,
)
from thinchat.openai_client import OpenAICompatibleClient
from thinchat.providers import PROVIDERS, make_client

__all__ = [
    "PROVIDERS",
    "Capability",
    "ClaudeClient",
    "Client",
    "LLMError",
    "OpenAICompatibleClient",
    "Provider",
    "ProviderUnavailableError",
    "ThinchatError",
    "UnknownProviderError",
    "UnsupportedError",
    "make_client",
]

try:
    __version__ = version("thinchat")   # single source of truth: the installed metadata
except PackageNotFoundError:            # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
