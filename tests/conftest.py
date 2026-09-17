"""Test isolation shared by the whole suite.

Now that key resolution reads a file store (``~/.config/thinchat/credentials.json`` via
credbox), a test that calls ``get_api_key`` would otherwise see the developer's real store or
a stray environment key -- non-deterministic, and different in CI. The autouse fixture points
the store at a fresh temp directory and clears the provider environment variables, so every
test starts from an empty, private credential state.
"""

import os

import pytest

# thinchat.keys binds its store via Credentials.for_app, which reads THINCHAT_STORE_APP /
# THINCHAT_NAMESPACE the first time a key call builds the store. Clear a developer's shell values
# here (before any test touches it) so the suite exercises the standalone binding, not a redirect.
os.environ.pop("THINCHAT_STORE_APP", None)
os.environ.pop("THINCHAT_NAMESPACE", None)

# The API-key env vars, plus OLLAMA_HOST: ollama's endpoint is resolved from the environment, so
# a developer's exported OLLAMA_HOST would otherwise bleed into the default-endpoint tests.
_PROVIDER_ENV_VARS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "CLAUDE_API_KEY", "OLLAMA_HOST")


@pytest.fixture(autouse=True)
def isolate_api_key_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
    # `_get_credentials` caches one binding for the process (`@lru_cache`); a test that redirects
    # it with THINCHAT_STORE_APP / THINCHAT_NAMESPACE would otherwise leave that redirect cached
    # past the monkeypatched env that built it. Clear it around each test so the redirect is local.
    from thinchat.keys import _get_credentials

    _get_credentials.cache_clear()
    yield
    _get_credentials.cache_clear()
