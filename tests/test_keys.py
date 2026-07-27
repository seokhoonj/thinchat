"""Key resolution from the environment, and the explicit-override precedence."""

from tests.fakes import install_openai
from thinchat import make_client
from thinchat.keys import get_api_key


def test_reads_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_api_key("openai") == "sk-test"


def test_missing_key_is_none(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_api_key("gemini") is None


def test_ollama_has_no_key_name():
    assert get_api_key("ollama") is None   # not in ENV_BY_PROVIDER -- a local server needs none


def test_an_explicit_key_overrides_the_environment(monkeypatch):
    seen: dict[str, object] = {}
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    install_openai(monkeypatch, client_capture=seen)
    make_client("openai", api_key="explicit")   # explicit wins over the set env var
    assert seen["api_key"] == "explicit"
