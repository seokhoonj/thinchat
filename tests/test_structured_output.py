"""Parsing a model's reply into a JSON object, and the instruction that asks for one."""

import pytest

from thinchat.errors import LLMError
from thinchat.structured_output import make_json_instruction, parse_json


def test_parses_a_plain_object():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_strips_json_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_pulls_the_object_out_of_surrounding_prose():
    assert parse_json('Sure! {"a": 1} hope that helps.') == {"a": 1}


def test_empty_reply_raises():
    with pytest.raises(LLMError):
        parse_json("   ")


def test_a_reply_with_no_object_raises():
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_a_reply_with_no_braces_raises():
    # "[1,2,3]" has no `{`, so it fails the no-object guard (the span never reaches json.loads).
    with pytest.raises(LLMError):
        parse_json("[1, 2, 3]")


def test_bare_empty_braces_in_prose_parse_as_an_empty_object():
    # Documented behaviour of the greedy outermost-brace span: a stray {} yields {}.
    assert parse_json("Use {} for an empty set.") == {}


def test_two_objects_in_one_reply_raise():
    # The greedy span runs first `{` to last `}`, swallowing the prose between -> invalid JSON.
    with pytest.raises(LLMError):
        parse_json('{"a": 1} and also {"b": 2}')


def test_instruction_carries_the_schema():
    text = make_json_instruction({"type": "object", "properties": {"x": {"type": "string"}}})
    assert "JSON" in text and "x" in text


# --- hostile replies: json.loads raises more than JSONDecodeError --------------------------

def test_a_giant_int_literal_raises_llmerror_not_valueerror():
    # An integer literal past ~4300 digits makes json.loads raise a *bare* ValueError (not a
    # JSONDecodeError), which used to escape the JSONDecodeError-only handler and break the
    # LLMError contract. It must now surface as an LLMError.
    reply = '{"n": ' + "1" * 4400 + "}"
    with pytest.raises(LLMError):
        parse_json(reply)


def test_a_recursion_error_from_the_parser_becomes_an_llmerror(monkeypatch):
    # Deeply nested JSON makes json.loads raise a bare RecursionError, which used to escape the
    # JSONDecodeError-only handler. We do NOT feed real deep input here: past a few thousand
    # levels the C scanner can overflow the C stack and *crash* the interpreter (a SIGSEGV, not
    # a catchable RecursionError), so the test would be a segfault rather than an assertion.
    # Inject the RecursionError at the parse call instead, which exercises the exact handler.
    def blow_the_stack(_text):
        raise RecursionError("maximum recursion depth exceeded while decoding JSON")

    monkeypatch.setattr("thinchat.structured_output.json.loads", blow_the_stack)
    with pytest.raises(LLMError):
        parse_json('{"a": 1}')


def test_error_message_neutralizes_control_characters():
    # A model reply echoed into the error message must not smuggle ANSI escapes or carriage
    # returns into a terminal/log line; C0 controls are replaced with spaces.
    with pytest.raises(LLMError) as exc_info:
        parse_json("prose with an \x1b[31m escape and \r return, no object")
    message = str(exc_info.value)
    assert "\x1b" not in message and "\r" not in message


def test_a_non_object_json_value_raises():
    # A well-formed JSON value that is not an object (the span parses to a list) is refused.
    assert parse_json('{"a": 1}') == {"a": 1}   # sanity: an object still parses
    with pytest.raises(LLMError):
        parse_json('prefix {"a": 1} , {"b": 2} suffix')   # greedy span -> invalid JSON


def test_an_unencodable_schema_raises_llmerror():
    # A schema json.dumps cannot encode (a non-serializable value) surfaces as an LLMError on
    # the structured-output path, not a raw TypeError, and the message never echoes the schema.
    with pytest.raises(LLMError) as exc_info:
        make_json_instruction({"bad": object()})
    assert "<object object" not in str(exc_info.value)   # the schema's repr is not echoed back
