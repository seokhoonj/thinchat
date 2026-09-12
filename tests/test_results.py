"""The Completion result object: a str subclass carrying the reply's metadata."""

from thinchat import Completion, Usage


def test_completion_is_a_string_carrying_metadata():
    completion = Completion("hello", finish_reason="stop", truncated=False,
                            usage=Usage(input_tokens=3, output_tokens=5), model="m")
    assert isinstance(completion, str) and completion == "hello"     # behaves as the text
    assert completion.upper() == "HELLO" and completion + "!" == "hello!"
    assert completion.finish_reason == "stop" and completion.truncated is False
    assert completion.usage is not None and completion.usage.input_tokens == 3
    assert completion.model == "m"


def test_completion_defaults_to_empty_metadata():
    completion = Completion("x")
    assert completion.finish_reason is None and completion.truncated is False
    assert completion.usage is None and completion.model is None


def test_completion_repr_shows_the_text_and_metadata():
    rendered = repr(Completion("hi", finish_reason="stop", model="m"))
    assert rendered.startswith("Completion(") and "'hi'" in rendered
    assert "finish_reason='stop'" in rendered and "model='m'" in rendered
