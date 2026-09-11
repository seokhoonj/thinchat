"""Turn a model's text reply into a JSON object, and ask for one in the first place.

Structured output has two halves. A client that supports it natively (OpenAI's
``response_format``) constrains the reply to a JSON object at the API -- not to the schema
itself -- while one that does not is steered by appending the schema to the prompt. Either
way schema conformance is best-effort, and the reply is text that must be
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
    output: reply as JSON matching this schema, nothing else.

    Raises:
        LLMError: ``schema`` cannot be encoded as JSON (not serializable, self-referential,
            or nested past the recursion limit) -- a bad schema surfaces on the parse path
            like any other structured-output failure, not as a raw ``TypeError``/``ValueError``.
    """
    try:
        encoded = json.dumps(schema)
    except (ValueError, TypeError, RecursionError) as err:
        # Content-free: name the failure, never echo the schema back into the message.
        raise LLMError("the structured-output schema could not be encoded as JSON") from err
    return (
        "Reply with a single JSON object matching this JSON Schema, and nothing else -- "
        f"no prose, no code fences:\n{encoded}"
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
        raise LLMError(f"structured reply held no JSON object: {_snippet(text)}")
    # json.loads does not only raise JSONDecodeError: an integer literal past ~4300 digits
    # raises a bare ValueError, and deeply nested brackets a RecursionError -- both from a
    # hostile-but-well-fenced reply. Catch the broad ValueError (JSONDecodeError's own base)
    # and RecursionError so neither escapes the package's LLMError contract.
    try:
        parsed = json.loads(text[start : end + 1])
    except (ValueError, RecursionError) as err:
        raise LLMError(f"structured reply was not valid JSON: {_snippet(text)}") from err
    if not isinstance(parsed, dict):
        raise LLMError("structured reply was JSON but not an object")
    return parsed


def neutralize_controls(text: str) -> str:
    """Replace C0 and C1 control characters (plus DEL) with spaces, so an untrusted string
    echoed into an error message cannot smuggle ANSI escapes or carriage returns into a
    terminal or log line. C1 (U+0080-U+009F) is included because U+009B is CSI -- the
    single-character form of ``ESC[`` that terminals honour in UTF-8 mode -- so dropping it
    would leave the very escape sequence this guards against."""
    return "".join(" " if ch < " " or "\x7f" <= ch <= "\x9f" else ch for ch in text)


def _snippet(text: str) -> str:
    """The first 120 characters of ``text`` with control characters neutralized (see
    ``neutralize_controls``), for echoing an untrusted reply into an error message safely."""
    return neutralize_controls(text[:120])


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
