"""Key resolution (override > env > stored file) and the credential-store API."""

import json
import os

import pytest
from credbox import CredBoxError, config_dir

from tests.fakes import install_openai
from thinchat import make_client
from thinchat.errors import CredentialStoreError, UnknownProviderError, UnsupportedError
from thinchat.keys import (
    _STORE_APP,
    get_api_key,
    set_api_key,
    stored_providers,
    unset_api_key,
)


def _store_path():
    return config_dir(_STORE_APP) / "credentials.json"   # the app name keys.py actually writes under


def _write_store(contents):
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def test_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert (k := get_api_key("openai")) is not None and k.reveal() == "sk-test"


def test_missing_key_is_none():
    assert get_api_key("gemini") is None


def test_ollama_has_no_key_name():
    assert get_api_key("ollama") is None   # known, but a local server needs no key


def test_get_rejects_an_unknown_provider():
    with pytest.raises(UnknownProviderError):
        get_api_key("bogus")   # a typo is an error, not a misleading "no key"


def test_an_explicit_key_overrides_the_environment(monkeypatch):
    client_arguments: dict[str, object] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    install_openai(monkeypatch, client_capture=client_arguments)
    make_client("openai", api_key="explicit")   # explicit wins over the set env var
    assert client_arguments["api_key"] == "explicit"


def test_explicit_api_key_does_not_read_the_store(monkeypatch):
    # The pure-library invariant: api_key= short-circuits before any file read. A malformed
    # store would raise CredentialStoreError if it were read, so a clean construction with the
    # explicit key proves the store was never touched.
    client_arguments: dict[str, object] = {}
    _write_store("{ this is not valid json")
    monkeypatch.setenv("OPENAI_API_KEY", "hostile-env")
    install_openai(monkeypatch, client_capture=client_arguments)
    make_client("openai", api_key="explicit")
    assert client_arguments["api_key"] == "explicit"


def test_stores_a_key_then_reads_it_back():
    set_api_key("claude", value="sk-stored")
    assert (k := get_api_key("claude")) is not None and k.reveal() == "sk-stored"


def test_storing_over_an_existing_key_overwrites():
    set_api_key("openai", value="first")
    set_api_key("openai", value="second")
    assert (k := get_api_key("openai")) is not None and k.reveal() == "second"
    assert stored_providers() == ["openai"]   # listed once, not twice


def test_environment_beats_the_stored_key(monkeypatch):
    set_api_key("openai", value="from-store")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert (k := get_api_key("openai")) is not None and k.reveal() == "from-env"


def test_explicit_override_beats_environment_and_store(monkeypatch):
    set_api_key("openai", value="from-store")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert (k := get_api_key("openai", override="explicit")) is not None and k.reveal() == "explicit"


def test_blank_environment_falls_through_to_the_store(monkeypatch):
    set_api_key("gemini", value="from-store")
    monkeypatch.setenv("GEMINI_API_KEY", "")   # a blank env value is treated as absent
    assert (k := get_api_key("gemini")) is not None and k.reveal() == "from-store"


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


def test_unset_rejects_ollama_which_needs_no_key():
    with pytest.raises(UnsupportedError):
        unset_api_key("ollama")


def test_set_rejects_a_blank_value():
    with pytest.raises(ValueError):
        set_api_key("claude", value="   ")
    assert stored_providers() == []   # a blank key that would resolve as absent is not stored


@pytest.mark.skipif(os.name != "posix", reason="0600 file mode is a POSIX concept")
def test_stored_key_file_is_owner_readable_only():
    set_api_key("claude", value="sk-secret")
    mode = _store_path().stat().st_mode & 0o777
    assert mode == 0o600


@pytest.mark.parametrize(
    "operation",
    [
        lambda: get_api_key("claude"),
        lambda: set_api_key("claude", value="sk-x"),
        lambda: unset_api_key("claude"),
        lambda: stored_providers(),
    ],
    ids=["get", "set", "unset", "stored_providers"],
)
def test_a_malformed_store_raises_credential_store_error(operation):
    _write_store(json.dumps(["not", "an", "object"]))   # a JSON array, not the expected object
    with pytest.raises(CredentialStoreError):
        operation()


def test_set_on_a_malformed_store_preserves_the_file():
    original = json.dumps(["not", "an", "object"])
    _write_store(original)
    with pytest.raises(CredentialStoreError):
        set_api_key("claude", value="sk-x")
    assert _store_path().read_text() == original   # a failed write must not clobber the file


def test_a_malformed_store_error_never_echoes_the_file_contents():
    # A store-read failure must not surface the file's bytes -- which may hold a real key -- in
    # the error or anywhere down its cause chain.
    secret = "sk-SENTINEL-IN-FILE-0000"
    _write_store('{"CLAUDE_API_KEY": "' + secret + '", MALFORMED')
    with pytest.raises(CredentialStoreError) as exc_info:
        get_api_key("claude")
    error: BaseException | None = exc_info.value
    while error is not None:
        assert secret not in str(error)
        error = error.__cause__


def test_a_store_write_failure_never_adds_the_key_to_the_error(monkeypatch):
    # thinchat wraps a write failure with the provider name only, never the value it was
    # handed; the credbox cause is secret-safe by contract, so the chain is not re-scrubbed.
    secret = "sk-VALUE-IN-HAND-1111"

    def fail_write(*args, **kwargs):
        raise CredBoxError("backend write failed")

    monkeypatch.setattr("thinchat.keys._store.set", fail_write)
    with pytest.raises(CredentialStoreError) as exc_info:
        set_api_key("claude", value=secret)
    assert secret not in str(exc_info.value)
