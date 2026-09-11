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
    assert (k := get_api_key("claude")) is not None and k.reveal() == "sk-entered"
    assert "stored the API key for claude" in capsys.readouterr().out


def test_set_strips_surrounding_whitespace(monkeypatch):
    _answer_prompt_with(monkeypatch, "  sk-padded  ")
    main(["set", "openai"])
    assert (k := get_api_key("openai")) is not None and k.reveal() == "sk-padded"


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


def test_get_masks_a_key_resolved_from_the_environment(monkeypatch, capsys):
    secret = "sk-env-1234567890ABCDEFGH"
    monkeypatch.setenv("CLAUDE_API_KEY", secret)   # resolved from env, not the file store
    assert main(["get", "claude"]) == 0
    shown = capsys.readouterr().out
    assert secret not in shown
    assert "sk-e" in shown   # masked, edges only


def test_get_on_a_missing_key_reports_no_key(capsys):
    assert main(["get", "gemini"]) == 0
    assert "no key for gemini" in capsys.readouterr().out


def test_list_shows_which_providers_are_set(capsys):
    set_api_key("claude", value="sk-c")
    assert main(["list"]) == 0
    listed = capsys.readouterr().out
    assert "claude" in listed and "set" in listed
    assert "not set" in listed   # openai / gemini are unset


def test_unset_removes_the_stored_key(capsys):
    set_api_key("openai", value="sk-o")
    assert main(["unset", "openai"]) == 0
    assert get_api_key("openai") is None
    assert "removed the stored key for openai" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["set", "get", "unset"])
def test_an_unknown_provider_reports_an_error_and_exits_one(command, monkeypatch, capsys):
    _answer_prompt_with(monkeypatch, "sk-x")   # set validates before prompting; patch is a safety net
    assert main([command, "bogus"]) == 1
    assert "thinchat:" in capsys.readouterr().err


@pytest.mark.parametrize("provider", ["bogus", "ollama"])   # a typo, and a known-but-keyless one
def test_set_validates_the_provider_before_prompting(provider, monkeypatch):
    def must_not_prompt(prompt=""):
        raise AssertionError("prompted for a key before validating the provider")
    monkeypatch.setattr(getpass, "getpass", must_not_prompt)
    assert main(["set", provider]) == 1   # rejected without ever asking for a key


@pytest.mark.parametrize(
    "length, expected",
    # credbox's Secret masking (edge=4): the edges appear only once the value is long enough
    # that they leave a hidden middle (len >= 4*edge = 16); anything shorter is a fixed "***"
    # that reveals neither the edges nor the length.
    [(15, "***"), (16, "kkkk...kkkk")],
)
def test_get_reveals_edges_only_above_the_mask_threshold(length, expected, capsys):
    set_api_key("claude", value="k" * length)
    assert main(["get", "claude"]) == 0
    assert capsys.readouterr().out.strip() == expected


def test_version_prints_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_stored_key_never_appears_in_any_cli_output(monkeypatch, capsys):
    """The key value must not reach stdout or stderr on any CLI path, including `set`."""
    secret = "sk-DO-NOT-LEAK-THIS-0987654321"
    _answer_prompt_with(monkeypatch, secret)
    for argv in (["set", "gemini"], ["get", "gemini"], ["list"], ["unset", "gemini"]):
        main(argv)
        captured = capsys.readouterr()
        assert secret not in captured.out
        assert secret not in captured.err
