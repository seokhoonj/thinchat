"""Offline contract checks against the REAL installed vendor SDKs.

The rest of the suite runs against hand-built fakes (tests/fakes.py), which stay frozen while
the real ``openai`` / ``anthropic`` SDKs evolve -- so a vendor rename or removal of an attribute
the clients read would pass every faked test yet break real usage. These assert, on the real
modules and offline (constructing a client stores the key but sends nothing), that the exact
call surfaces the clients walk still exist, so an SDK upgrade that breaks them fails here.
"""

import anthropic
import openai


def test_openai_sdk_exposes_the_call_surfaces_the_client_uses():
    assert issubclass(openai.RateLimitError, openai.OpenAIError)   # error mapping depends on this
    client = openai.OpenAI(api_key="x")            # constructs offline; no request is sent
    assert callable(client.chat.completions.create)
    assert callable(client.embeddings.create)
    aclient = openai.AsyncOpenAI(api_key="x")
    assert callable(aclient.chat.completions.create)
    assert callable(aclient.embeddings.create)


def test_anthropic_sdk_exposes_the_call_surfaces_the_client_uses():
    assert issubclass(anthropic.RateLimitError, anthropic.AnthropicError)
    client = anthropic.Anthropic(api_key="x")      # constructs offline; no request is sent
    assert callable(client.messages.create)
    assert callable(client.messages.stream)
    aclient = anthropic.AsyncAnthropic(api_key="x")
    assert callable(aclient.messages.create)
    assert callable(aclient.messages.stream)


def test_openai_response_types_expose_the_attributes_the_client_reads():
    # The completion-metadata reads (_completion_from_chat) and the embedding alignment use
    # getattr, so a renamed field would silently null the metadata with no failing test. Pin the
    # real response-type field names, so an SDK rename fails HERE instead of shipping green.
    from openai.types import CompletionUsage, Embedding
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    assert {"choices", "model", "usage"} <= set(ChatCompletion.model_fields)
    assert {"finish_reason", "message"} <= set(Choice.model_fields)
    assert "content" in ChatCompletionMessage.model_fields
    assert {"prompt_tokens", "completion_tokens"} <= set(CompletionUsage.model_fields)
    assert {"embedding", "index"} <= set(Embedding.model_fields)


def test_anthropic_response_types_expose_the_attributes_the_client_reads():
    from anthropic.types import Message, TextBlock, Usage
    assert {"content", "stop_reason", "usage", "model"} <= set(Message.model_fields)
    assert {"input_tokens", "output_tokens"} <= set(Usage.model_fields)
    assert {"type", "text"} <= set(TextBlock.model_fields)
