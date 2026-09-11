"""Regressions for the three key-leak vectors a red-team review found, plus the input
hardening around them. Each test provokes the exact path that used to leak and asserts the
key is now unreachable, or that the endpoint is pinned so no key can be redirected.

The three vectors:
  1. runtime error path -- the SDK error was chained (``raise ... from err``), keeping the key
     in ``err.request.headers`` reachable. Covered in test_secret_scrubbing.py (severance).
  2. construction path -- a constructor failure (e.g. a malformed proxy httpx rejects) escaped
     with the revealed key live in the SDK's ``__init__`` frame. Covered here.
  3. base_url env -- ``base_url=None`` let the SDK read ``OPENAI_BASE_URL`` /
     ``ANTHROPIC_BASE_URL`` and ship the resolved key to that host. Covered here.
"""

import pytest

from tests.fakes import FakeAnthropicError, FakeOpenAIError, install_anthropic, install_openai
from thinchat import make_client
from thinchat.errors import ProviderUnavailableError

_KEY = "sk-super-secret-KEY-should-never-leak-42"


# --- vector 2: a constructor failure must not escape with the key on its frame -------------

def test_openai_construction_failure_severs_and_hides_the_key(monkeypatch):
    # A malformed ALL_PROXY makes httpx reject the transport URL from inside OpenAI.__init__,
    # whose frame holds the revealed key. build_sdk_client converts it to a content-free
    # ProviderUnavailableError raised from a clean frame -- chain severed, key absent.
    boom = FakeOpenAIError(f"Invalid proxy URL for key {_KEY}")
    install_openai(monkeypatch, client_error=boom)
    with pytest.raises(ProviderUnavailableError) as exc_info:
        make_client("openai", api_key=_KEY)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert _KEY not in str(exc_info.value)


def test_claude_construction_failure_severs_and_hides_the_key(monkeypatch):
    boom = FakeAnthropicError(f"Invalid proxy URL for key {_KEY}")
    install_anthropic(monkeypatch, client_error=boom)
    with pytest.raises(ProviderUnavailableError) as exc_info:
        make_client("claude", api_key=_KEY)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert _KEY not in str(exc_info.value)


async def test_openai_async_construction_failure_severs_the_key(monkeypatch):
    # The async client is built lazily on first async use; its constructor frame holds the key
    # the same way, so the async build path must sever too.
    boom = FakeOpenAIError(f"async proxy rejected for key {_KEY}")
    install_openai(monkeypatch)                    # sync client builds fine
    client = make_client("openai", api_key=_KEY)
    install_openai(monkeypatch, client_error=boom)  # re-install so the ASYNC constructor is the one that fails
    with pytest.raises(ProviderUnavailableError) as exc_info:
        await client.acomplete("hi")
    assert exc_info.value.__cause__ is None and exc_info.value.__context__ is None
    assert _KEY not in str(exc_info.value)


# --- vector 3: base_url is pinned, so the SDK never reads its *_BASE_URL env ----------------

def test_openai_base_url_is_pinned_over_a_hostile_env(monkeypatch):
    # An attacker who can write the environment (but not read the 0600 store) sets
    # OPENAI_BASE_URL to their host. With base_url pinned, the constructor gets the official
    # URL, not the attacker's -- the SDK never sees the env var.
    monkeypatch.setenv("OPENAI_BASE_URL", "https://evil.example/v1")
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("openai", api_key=_KEY)
    assert seen["base_url"] == "https://api.openai.com/v1"


def test_claude_base_url_is_pinned_over_a_hostile_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example")
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, client_capture=seen)
    make_client("claude", api_key=_KEY)
    assert seen["base_url"] == "https://api.anthropic.com"


def test_an_explicit_base_url_wins(monkeypatch):
    # A caller pointing at a real gateway passes base_url= as a code decision; that must reach
    # the SDK unchanged (the pin only supplies the default when none is passed).
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("openai", api_key=_KEY, base_url="https://gateway.internal/v1")
    assert seen["base_url"] == "https://gateway.internal/v1"


def test_gemini_base_url_default_is_pinned_not_none(monkeypatch):
    # Gemini's default is its own compat endpoint (not the openai one), and never None.
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("gemini", api_key=_KEY)
    assert seen["base_url"] is not None
    assert "generativelanguage.googleapis.com" in str(seen["base_url"])


def test_ollama_base_url_defaults_to_the_local_host(monkeypatch):
    # ollama is keyless (no key to redirect); its default endpoint comes from OLLAMA_HOST.
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("ollama")
    assert seen["base_url"] is not None
    assert "11434" in str(seen["base_url"])


# --- the embed bare-str trap: str is a Sequence[str] of characters -------------------------

def test_embed_rejects_a_bare_string(monkeypatch):
    # embed("hello") would silently embed one vector per character; the guard is a TypeError,
    # a caller bug, raised before any API call.
    install_openai(monkeypatch)
    client = make_client("openai", api_key=_KEY)
    with pytest.raises(TypeError):
        # mypy cannot catch this -- a str *is* a Sequence[str] -- which is exactly why the
        # runtime guard exists; passing a bare str is a valid call to the type checker.
        client.embed("hello")


async def test_aembed_rejects_a_bare_string(monkeypatch):
    install_openai(monkeypatch)
    client = make_client("openai", api_key=_KEY)
    with pytest.raises(TypeError):
        await client.aembed("hello")


def test_embed_accepts_a_list(monkeypatch):
    install_openai(monkeypatch, vectors=([0.1, 0.2],))
    client = make_client("openai", api_key=_KEY)
    assert client.embed(["hello"]) == [[0.1, 0.2]]


def test_embed_of_no_texts_makes_no_call(monkeypatch):
    # An empty list is zero vectors, not an error, and must not hit the SDK.
    install_openai(monkeypatch, error=FakeOpenAIError("should not be called"))
    client = make_client("openai", api_key=_KEY)
    assert client.embed([]) == []
