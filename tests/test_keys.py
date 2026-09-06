"""Key resolution (override > env > stored file) and the credential-store API."""

import json
import os

import pytest
from xdg_kit import config_dir

from tests.fakes import install_openai
from thinchat import make_client
from thinchat.errors import CredentialStoreError, UnknownProviderError, UnsupportedError
from thinchat.keys import (
    get_api_key,
    set_api_key,
    stored_providers,
    unset_api_key,
)


def _store_path():
    return config_dir("thinchat") / "credentials.json"


def test_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_api_key("openai") == "sk-test"


def test_missing_key_is_none():
    assert get_api_key("gemini") is None


def test_ollama_has_no_key_name():
    assert get_api_key("ollama") is None   # not in ENV_BY_PROVIDER -- a local server needs none


def test_an_explicit_key_overrides_the_environment(monkeypatch):
    seen: dict[str, object] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    install_openai(monkeypatch, client_capture=seen)
    make_client("openai", api_key="explicit")   # explicit wins over the set env var
    assert seen["api_key"] == "explicit"


def test_stores_a_key_then_reads_it_back():
    set_api_key("claude", value="sk-stored")
    assert get_api_key("claude") == "sk-stored"


def test_environment_beats_the_stored_key(monkeypatch):
    set_api_key("openai", value="from-store")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert get_api_key("openai") == "from-env"


def test_explicit_override_beats_environment_and_store(monkeypatch):
    set_api_key("openai", value="from-store")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert get_api_key("openai", override="explicit") == "explicit"


def test_blank_environment_falls_through_to_the_store(monkeypatch):
    set_api_key("gemini", value="from-store")
    monkeypatch.setenv("GEMINI_API_KEY", "")   # a blank env value is treated as absent
    assert get_api_key("gemini") == "from-store"


def test_stored_providers_reflects_set_and_unset():
    assert stored_providers() == []
    set_api_key("claude", value="sk-c")
    set_api_key("openai", value="sk-o")
    assert stored_providers() == ["openai", "claude"]   # ENV_BY_PROVIDER order, not set order
    unset_api_key("claude")
    assert stored_providers() == ["openai"]


def test_unset_is_a_no_op_when_absent():
    unset_api_key("gemini")   # nothing stored -- must not raise
    assert get_api_key("gemini") is None


def test_set_rejects_an_unknown_provider():
    with pytest.raises(UnknownProviderError):
        set_api_key("bogus", value="sk-x")


def test_set_rejects_ollama_which_needs_no_key():
    with pytest.raises(UnsupportedError):
        set_api_key("ollama", value="sk-x")


@pytest.mark.skipif(os.name != "posix", reason="0600 file mode is a POSIX concept")
def test_stored_key_file_is_owner_readable_only():
    set_api_key("claude", value="sk-secret")
    mode = _store_path().stat().st_mode & 0o777
    assert mode == 0o600


def test_a_malformed_store_raises_credential_store_error():
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(["not", "an", "object"]))   # a JSON array, not the expected object
    with pytest.raises(CredentialStoreError):
        get_api_key("claude")
