"""What a completion verb returns.

``complete`` / ``acomplete`` return a ``Completion`` -- a real ``str`` (so it prints, compares,
concatenates, and slices exactly like the text it holds, and every caller that just wanted the
text is unchanged) that ALSO carries the reply's metadata: the provider's stop reason, whether
the reply was cut off at the token cap, the token usage, and the model that served it. The bare
text stays the primary value; the metadata is one attribute away for a caller that needs it (e.g.
to detect a truncated reply, which a bare string cannot express).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Completion", "Usage"]


def _as_int(value: object) -> int | None:
    """``value`` if it is a real int (a bool is not a token count), else ``None`` -- for reading a
    token count off a vendor response defensively."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class Usage:
    """Token counts for one request, each ``None`` when the provider did not report it."""

    input_tokens:  int | None = None
    output_tokens: int | None = None


class Completion(str):
    """The reply text from ``complete``, as a ``str`` subclass carrying the reply's metadata.

    Use it as a string for the common case (``print(llm.complete(p))``); read the attributes when
    you need them:

    - ``finish_reason``: the provider's own stop reason, verbatim (OpenAI ``finish_reason`` --
      ``stop`` / ``length`` / ...; Anthropic ``stop_reason`` -- ``end_turn`` / ``max_tokens`` /
      ...), or ``None`` when the provider did not report one.
    - ``truncated``: ``True`` when the reply was cut off at the token cap -- normalized across
      providers (OpenAI ``length`` / Anthropic ``max_tokens``). A bare string cannot tell a
      complete reply from a truncated one; this can.
    - ``usage``: input/output token counts (a ``Usage``), or ``None`` when unreported.
    - ``model``: the model id that actually served the request, or ``None``.
    """

    finish_reason: str | None
    truncated:     bool
    usage:         Usage | None
    model:         str | None

    def __new__(
        cls, text: str, *, finish_reason: str | None = None, truncated: bool = False,
        usage: Usage | None = None, model: str | None = None,
    ) -> Completion:
        self = super().__new__(cls, text)
        self.finish_reason = finish_reason
        self.truncated     = truncated
        self.usage         = usage
        self.model         = model
        return self

    def __repr__(self) -> str:
        return (f"Completion({str.__repr__(self)}, finish_reason={self.finish_reason!r}, "
                f"truncated={self.truncated!r}, model={self.model!r})")
