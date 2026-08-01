"""The public package surface exposes the documented exception hierarchy."""

from thinchat import LLMError, RateLimitError


def test_rate_limit_error_is_exported_as_an_llm_error():
    assert issubclass(RateLimitError, LLMError)
