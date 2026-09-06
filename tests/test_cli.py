"""The `thinchat` CLI: storing, showing (masked), listing, and removing provider keys.

The credential store is isolated per test by the autouse fixture in conftest.py, so these
run against a fresh temp store, never the developer's real one.
"""

import getpass

import pytest

from thinchat import __version__
from thinchat.cli import main
from thinchat.keys import get_api_key, set_api_key


def _answer_prompt_with(monkeypatch, value):
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": value)


def test_set_stores_the_entered_key(monkeypatch, capsys):
    _answer_prompt_with(monkeypatch, "sk-entered")
    exit_code = main(["set", "claude"])
    assert exit_code == 0
    assert get_api_key("claude") == "sk-entered"
    assert "stored the API key for claude" in capsys.readouterr().out


def test_set_strips_surrounding_whitespace(monkeypatch):
    _answer_prompt_with(monkeypatch, "  sk-padded  ")
    main(["set", "openai"])
    assert get_api_key("openai") == "sk-padded"


def test_set_with_blank_input_is_a_usage_error(monkeypatch, capsys):
    _answer_prompt_with(monkeypatch, "   ")
    assert main(["set", "claude"]) == 2
    assert get_api_key("claude") is None
    assert "no key provided" in capsys.readouterr().err


def test_set_on_closed_stdin_is_a_usage_error(monkeypatch, capsys):
    def raise_eof(prompt=""):
        raise EOFError
    monkeypatch.setattr(getpass, "getpass", raise_eof)
    assert main(["set", "claude"]) == 2
    assert "stdin closed" in capsys.readouterr().err


def test_get_masks_the_key_and_never_prints_it_whole(capsys):
    full_key = "sk-1234567890ABCDEFGH"
    set_api_key("claude", value=full_key)
    assert main(["get", "claude"]) == 0
    shown = capsys.readouterr().out
    assert full_key not in shown
    assert "sk-1" in shown and "EFGH" in shown   # only the edges are revealed


def test_get_on_a_missing_key_reports_no_key(capsys):
    assert main(["get", "gemini"]) == 0
    assert "no key for gemini" in capsys.readouterr().out


def test_list_shows_which_providers_are_set(capsys):
    set_api_key("claude", value="sk-c")
    assert main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "claude" in listed and "set" in listed
    assert "not set" in listed   # openai / gemini are unset


def test_unset_removes_the_stored_key(monkeypatch, capsys):
    set_api_key("openai", value="sk-o")
    assert main(["unset", "openai"]) == 0
    assert get_api_key("openai") is None
    assert "removed the stored key for openai" in capsys.readouterr().out


def test_an_unknown_provider_reports_an_error_and_exits_one(monkeypatch, capsys):
    _answer_prompt_with(monkeypatch, "sk-x")
    assert main(["set", "bogus"]) == 1
    assert "thinchat:" in capsys.readouterr().err


def test_version_prints_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_stored_key_never_appears_in_any_cli_output(capsys):
    """PACKAGE_BOUNDARY Ch 12: the key value must not reach stdout or stderr on any path."""
    secret = "sk-DO-NOT-LEAK-THIS-0987654321"
    set_api_key("gemini", value=secret)
    for argv in (["get", "gemini"], ["list"], ["unset", "gemini"]):
        main(argv)
        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err
