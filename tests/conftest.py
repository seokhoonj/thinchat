"""Test isolation shared by the whole suite.

Now that key resolution reads a file store (``~/.config/thinchat/credentials.json`` via
xdg-kit), a test that calls ``get_api_key`` would otherwise see the developer's real store or
a stray environment key -- non-deterministic, and different in CI. The autouse fixture points
the store at a fresh temp directory and clears the provider environment variables, so every
test starts from an empty, private credential state.
"""

import pytest

_PROVIDER_ENV_VARS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "CLAUDE_API_KEY")


@pytest.fixture(autouse=True)
def isolate_credential_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    for env_var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)
