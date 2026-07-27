"""Turn a model's text reply into a JSON object, and ask for one in the first place.

Structured output has two halves. A client that supports it natively (OpenAI's
``response_format``) constrains the reply to a schema at the API; one that does not is
steered by appending the schema to the prompt. Either way the reply is text that must be
parsed, and models wrap JSON in prose or ```json fences often enough that a bare
``json.loads`` is too brittle -- so parsing strips the common wrappers first. Schema
validation stops at "it is a JSON object": deep validation would need a schema library,
and the caller, which owns the schema, checks its own fields.
"""

from __future__ import annotations

import json

from thinchat.errors import LLMError

__all__ = ["make_json_instruction", "parse_json"]


def make_json_instruction(schema: dict[str, object]) -> str:
    """The line appended to a system prompt to steer a client with no native structured
    output: reply as JSON matching this schema, nothing else."""
    return (
        "Reply with a single JSON object matching this JSON Schema, and nothing else -- "
        f"no prose, no code fences:\n{json.dumps(schema)}"
    )


def parse_json(reply: str) -> dict[str, object]:
    """Parse ``reply`` into a JSON object. Tolerates a ```json fence and a prose preamble or
    trailer around the object; it takes the outermost ``{`` .. ``}`` span, so brace
    characters *inside* the surrounding prose can widen the span and fail the parse (raising,
    never returning a wrong object). Returns the parsed object.

    Raises:
        LLMError: the reply is empty, holds no JSON object, or the JSON is not an object.
    """
    text = _unfence(reply).strip()
    if not text:
        raise LLMError("structured request returned an empty reply")
    # A model that adds a sentence around the object still parses if we take the outermost
    # brace span; a bare json.loads would fail on the surrounding prose.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"structured reply held no JSON object: {text[:120]}")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as err:
        raise LLMError(f"structured reply was not valid JSON: {text[:120]}") from err
    if not isinstance(parsed, dict):
        raise LLMError("structured reply was JSON but not an object")
    return parsed


def _unfence(text: str) -> str:
    """Strip a leading ```json / ``` fence and its closing ``` if present, else return
    ``text`` unchanged -- the one wrapper common enough to handle before brace-scanning."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    body = stripped[3:]
    if body[:4].lower() == "json":
        body = body[4:]
    return body.rsplit("```", 1)[0]
