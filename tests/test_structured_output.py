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
