"""The public package surface exposes the documented exception hierarchy and Secret contract."""

from thinchat import LLMError, RateLimitError


def test_rate_limit_error_is_exported_as_an_llm_error():
    assert issubclass(RateLimitError, LLMError)


def test_secret_is_the_credbox_type_with_its_masking_contract():
    # thinchat re-exports credbox.Secret as public API and documents .reveal()/masking as its own
    # contract; pin the identity and the mask/reveal behaviour at thinchat's OWN boundary, so a
    # credbox change to Secret is caught here, not only indirectly through the key/CLI tests.
    import credbox

    from thinchat import Secret
    assert Secret is credbox.Secret
    plaintext = "sk-abcdefghij0123456789"
    secret = Secret(plaintext)
    assert secret.reveal() == plaintext   # the only plaintext path
    assert plaintext not in str(secret)   # masked in str()
    assert plaintext not in repr(secret)  # and repr()
