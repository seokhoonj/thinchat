"""Key resolution (override > env > stored file) and the credential-store API."""

import json
import os
import subprocess
import sys

import pytest
from credbox import CredBoxError, Secret, config_dir

from tests.fakes import install_anthropic, install_openai
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


@pytest.mark.parametrize("raw", ["  sk-x  ", "sk-x", "   ", "", None])
def test_ollama_override_normalizes_like_the_credbox_keyed_tier(raw):
    # ollama has no credbox tier, so keys.get_api_key normalizes its override (strip +
    # blank-is-absent) locally; the keyed providers route the override THROUGH credbox. The two
    # rules never run on the same input, so nothing else can catch them drifting -- pin that they
    # agree, so a change to credbox's blank-is-absent rule that keys.py no longer mirrors fails here.
    ollama = get_api_key("ollama", override=raw)
    keyed  = get_api_key("openai", override=raw)   # goes through credbox's own override normalization
    if keyed is None:
        assert ollama is None
    else:
        assert ollama is not None and ollama.reveal() == keyed.reveal()


def test_get_rejects_an_unknown_provider():
    with pytest.raises(UnknownProviderError):
        get_api_key("bogus")   # a typo is an error, not a misleading "no key"


def test_an_explicit_key_overrides_the_environment(monkeypatch):
    client_arguments: dict[str, object] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    install_openai(monkeypatch, client_capture=client_arguments)
    make_client("openai", api_key="explicit")   # explicit wins over the set env var
    assert client_arguments["api_key"] == "explicit"


def test_get_api_key_returns_a_secret(monkeypatch):
    # The store returns a masking Secret, not a bare str -- pins the migrated return type so a
    # regression to a plain string (re-widening the plaintext) is caught.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-typed")
    assert isinstance(get_api_key("openai"), Secret)


def test_openai_sync_constructor_receives_the_revealed_key(monkeypatch):
    # The ONLY place the key may be revealed is the vendor SDK constructor. If a regression passed
    # the masked str(Secret) or the Secret object instead of .reveal()'d plaintext, auth would
    # silently break. Pin the plaintext at each of the four constructor sites + the ollama dummy.
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("openai", api_key="sk-openai-sync")
    assert seen["api_key"] == "sk-openai-sync"


async def test_openai_async_constructor_receives_the_revealed_key(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, aclient_capture=seen)
    client = make_client("openai", api_key="sk-openai-async")
    await client.acomplete("hi")   # first async use builds the async client
    assert seen["api_key"] == "sk-openai-async"


def test_claude_sync_constructor_receives_the_revealed_key(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, client_capture=seen)
    make_client("claude", api_key="sk-claude-sync")
    assert seen["api_key"] == "sk-claude-sync"


async def test_claude_async_constructor_receives_the_revealed_key(monkeypatch):
    seen: dict[str, object] = {}
    install_anthropic(monkeypatch, aclient_capture=seen)
    client = make_client("claude", api_key="sk-claude-async")
    await client.acomplete("hi")
    assert seen["api_key"] == "sk-claude-async"


def test_ollama_construction_uses_the_dummy_key(monkeypatch):
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("ollama")
    assert seen["api_key"] == "ollama"   # the non-secret placeholder, no real key needed


def test_ollama_with_a_blank_override_still_uses_the_dummy(monkeypatch):
    # Regression: a blank override must normalize to absent (not Secret("")), so ollama falls back
    # to the dummy rather than constructing the SDK with an empty key -- which the real OpenAI SDK
    # rejects with a foreign OpenAIError escaping make_client.
    seen: dict[str, object] = {}
    install_openai(monkeypatch, client_capture=seen)
    make_client("ollama", api_key="")
    assert seen["api_key"] == "ollama"


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


def test_a_blank_value_raises_blank_key_error_in_the_thinchat_hierarchy():
    # The blank-key refusal is a BlankKeyError -- both a ThinchatError (so `except ThinchatError`
    # catches it) and a ValueError (so the existing `except ValueError` still does).
    from thinchat.errors import BlankKeyError, ThinchatError
    with pytest.raises(BlankKeyError) as exc_info:
        set_api_key("openai", value="")
    assert isinstance(exc_info.value, ThinchatError)
    assert isinstance(exc_info.value, ValueError)


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
    # handed. Assert against the whole exception chain (__cause__/__context__), not just the
    # top message, so a future wrapper that echoed the value into the cause would be caught.
    secret = "sk-VALUE-IN-HAND-1111"

    class FailStore:
        def set(self, *args, **kwargs):
            raise CredBoxError("backend write failed")

    monkeypatch.setattr("thinchat.keys._get_credentials", lambda: FailStore())
    with pytest.raises(CredentialStoreError) as exc_info:
        set_api_key("claude", value=secret)

    seen: set[int] = set()
    pending: list[BaseException | None] = [exc_info.value]
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        assert secret not in str(error)
        pending += [error.__cause__, error.__context__]


# Both sibling override keys feed the same `for_app` validation, so a malformed value in either
# must surface identically -- as thinchat's own error, not credbox's.
@pytest.mark.parametrize("binding_var", ["THINCHAT_NAMESPACE", "THINCHAT_STORE_APP"])
def test_a_malformed_store_binding_surfaces_as_a_credential_store_error(monkeypatch, binding_var):
    # A bad binding makes credbox's `for_app` reject it. It must reach the caller as thinchat's
    # own CredentialStoreError (inside the documented catch surface), not as credbox's
    # InvalidAppNameError -- a foreign type callers were never told to catch.
    from thinchat.keys import _get_credentials

    monkeypatch.setenv(binding_var, "../evil")
    _get_credentials.cache_clear()
    with pytest.raises(CredentialStoreError):
        get_api_key("claude")


def test_importing_thinchat_does_not_crash_on_a_malformed_binding():
    # The store is built lazily, not at import, so a malformed THINCHAT_NAMESPACE must not abort
    # `import thinchat` -- the very host-embedding scenario `for_app` exists to serve. Prove both
    # halves in one process: the import succeeds, AND the failure is deferred to the first key
    # call, where it surfaces as thinchat's own CredentialStoreError.
    program = (
        "import thinchat\n"
        "from thinchat.errors import CredentialStoreError\n"
        "try:\n"
        "    thinchat.get_api_key('claude')\n"
        "except CredentialStoreError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit('binding error was not deferred to the first key call')\n"
    )
    env = {**os.environ, "THINCHAT_NAMESPACE": "../evil"}
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
