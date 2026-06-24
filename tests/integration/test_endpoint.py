"""
Integration tests for the mimOE inference endpoint.

These tests hit the live endpoint and require mimOE Studio to be running
with a model loaded. Run them with:

    pytest -m integration

They are excluded from the default test run (plain ``pytest``) to avoid
failures in environments where Studio is not available.
"""

from __future__ import annotations

import pytest
from openai import OpenAI


@pytest.mark.integration
def test_endpoint_returns_a_response(live_client: OpenAI, live_model: str) -> None:
    """A minimal chat completion should return a non-None response.

    This is the baseline connectivity test — if this fails, nothing else
    in the integration suite will pass.
    """
    response = live_client.chat.completions.create(
        model=live_model,
        messages=[{"role": "user", "content": "Say: OK"}],
        max_tokens=10,
    )
    assert response.choices[0].message.content is not None


@pytest.mark.integration
def test_response_content_is_non_empty(live_client: OpenAI, live_model: str) -> None:
    """The model should return a non-empty string for a simple prompt."""
    response = live_client.chat.completions.create(
        model=live_model,
        messages=[{"role": "user", "content": "Hello."}],
        max_tokens=20,
    )
    reply = response.choices[0].message.content
    assert reply is not None
    assert len(reply.strip()) > 0


@pytest.mark.integration
def test_system_prompt_is_accepted(live_client: OpenAI, live_model: str) -> None:
    """The endpoint should accept a system message without error."""
    response = live_client.chat.completions.create(
        model=live_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        max_tokens=20,
    )
    assert response.choices[0].message.content is not None


@pytest.mark.integration
def test_multi_turn_history_is_accepted(live_client: OpenAI, live_model: str) -> None:
    """The endpoint should accept a multi-turn message list without error.

    This validates the format that ``Conversation.messages`` produces, not
    whether the model correctly uses the context.
    """
    response = live_client.chat.completions.create(
        model=live_model,
        messages=[
            {"role": "user", "content": "My name is Alex."},
            {"role": "assistant", "content": "Nice to meet you, Alex."},
            {"role": "user", "content": "What is my name?"},
        ],
        max_tokens=20,
    )
    assert response.choices[0].message.content is not None


@pytest.mark.integration
def test_streaming_yields_chunks(live_client: OpenAI, live_model: str) -> None:
    """Streaming should yield at least one non-empty chunk."""
    stream = live_client.chat.completions.create(
        model=live_model,
        messages=[{"role": "user", "content": "Count: 1, 2, 3."}],
        stream=True,
        max_tokens=30,
    )
    chunks = [chunk.choices[0].delta.content or "" for chunk in stream]
    assert any(chunks), "Expected at least one non-empty chunk from the stream"


@pytest.mark.integration
def test_streaming_assembles_into_non_empty_reply(live_client: OpenAI, live_model: str) -> None:
    """All streamed chunks joined together should form a non-empty reply."""
    stream = live_client.chat.completions.create(
        model=live_model,
        messages=[{"role": "user", "content": "Say hello."}],
        stream=True,
        max_tokens=20,
    )
    full_reply = "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    assert len(full_reply.strip()) > 0
