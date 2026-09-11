"""Naming a provider: the roster, the factory, construction failures, and the routing
(base URL / key) each provider is wired to."""

import sys

import pytest

from tests.fakes import install_anthropic, install_openai
from thinchat import PROVIDERS, Client, make_client
from thinchat.errors import ProviderUnavailableError, UnknownProviderError


def test_roster_is_the_four_providers():
    assert set(PROVIDERS) == {"openai", "claude", "gemini", "ollama"}


def test_roster_order_is_stable():
    # PROVIDERS feeds user-facing error text ("choose one of ..."), so its order is a contract.
    assert PROVIDERS == ("claude", "openai", "gemini", "ollama")


def test_provider_literal_matches_the_runtime_roster():
    # The `Provider` type and the runtime roster are two hand-kept lists; keep them tied.
    from typing import get_args

    from thinchat.client import Provider
    assert set(get_args(Provider)) == set(PROVIDERS)


def test_provider_registries_are_coherent():
    # The Provider type, the factory roster, and the OpenAI-compat spec table are separate
    # registries kept in sync by import-time checks in providers.py / openai_client.py. Pin that
    # they currently agree, so a drift (and a broken guard) is caught as a test failure too.
    from typing import get_args

    from thinchat.client import Provider
    from thinchat.keys import ENV_BY_PROVIDER
    from thinchat.openai_client import _SPEC_BY_PROVIDER
    from thinchat.providers import _FACTORY_BY_PROVIDER
    assert set(_FACTORY_BY_PROVIDER) == set(get_args(Provider))
    assert set(_SPEC_BY_PROVIDER) <= set(_FACTORY_BY_PROVIDER)
    for provider, spec in _SPEC_BY_PROVIDER.items():
        assert spec.needs_key == (provider in ENV_BY_PROVIDER)


def test_constructed_clients_satisfy_the_client_protocol(monkeypatch):
    install_openai(monkeypatch)
    install_anthropic(monkeypatch)
    assert isinstance(make_client("ollama"), Client)
    assert isinstance(make_client("claude", api_key="k"), Client)


def test_unknown_provider_raises():
    with pytest.raises(UnknownProviderError):
        make_client("mistral")


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_make_an_openai_compatible_client(monkeypatch, provider):
    install_openai(monkeypatch)
    client = make_client(provider, api_key="k")
    assert client.model                       # a default model is set
    assert client.supports("embeddings")


def test_ollama_needs_no_key(monkeypatch):
    install_openai(monkeypatch)
    client = make_client("ollama")            # no key passed, none required
    assert client.model == "llama3.1"


def test_make_the_claude_client(monkeypatch):
    install_anthropic(monkeypatch)
    client = make_client("claude", api_key="k")
    assert not client.supports("embeddings")   # Anthropic has no embeddings API


def test_a_model_override_wins_over_the_default(monkeypatch):
    install_openai(monkeypatch)
    client = make_client("gemini", model="gemini-2.0-pro", api_key="k")
    assert client.model == "gemini-2.0-pro"


# --- capability matrix --------------------------------------------------------

@pytest.mark.parametrize("provider", ["openai", "gemini", "ollama"])
def test_openai_compatible_providers_support_every_capability(monkeypatch, provider):
    install_openai(monkeypatch)
    client = make_client(provider, api_key="k")
    assert all(client.supports(c) for c in ("completion", "streaming", "structured_output", "embeddings"))


def test_claude_supports_all_capabilities_except_embeddings(monkeypatch):
    install_anthropic(monkeypatch)
    client = make_client("claude", api_key="k")
    assert all(client.supports(c) for c in ("completion", "streaming", "structured_output"))
    assert not client.supports("embeddings")


# --- routing (base URL / key wired to the SDK) --------------------------------

def test_gemini_points_at_the_compatible_endpoint(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("gemini", api_key="k")
    assert seen["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert seen["api_key"] == "k"


def test_ollama_uses_a_dummy_key_and_v1_base_url(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("ollama")
    assert str(seen["base_url"]).endswith("/v1")
    assert seen["api_key"] == "ollama"   # ollama ignores it, but the SDK needs a non-empty key


def test_ollama_prepends_a_scheme_to_a_bare_host(monkeypatch):
    # A scheme-less OLLAMA_HOST (the ollama CLI accepts it) must gain http:// -- httpx cannot
    # use a scheme-relative base URL.
    seen: dict[str, object] = {}
    monkeypatch.setenv("OLLAMA_HOST", "myhost:11434")
    install_openai(monkeypatch, client_capture=seen)
    make_client("ollama")
    assert seen["base_url"] == "http://myhost:11434/v1"


# --- construction failures ----------------------------------------------------

@pytest.mark.parametrize("provider, env", [
    ("openai", "OPENAI_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("claude", "CLAUDE_API_KEY"),
])
def test_a_missing_key_raises(monkeypatch, provider, env):
    (install_anthropic if provider == "claude" else install_openai)(monkeypatch)
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(ProviderUnavailableError):
        make_client(provider)


def test_a_missing_openai_sdk_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)   # makes `from openai import ...` raise
    with pytest.raises(ProviderUnavailableError):
        make_client("openai", api_key="k")


def test_a_missing_anthropic_sdk_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ProviderUnavailableError):
        make_client("claude", api_key="k")
