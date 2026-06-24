"""
In-session conversation history.

Keeps the full message list in memory; history is lost on restart by design.
"""

from __future__ import annotations

import logging

from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class Conversation:
    """Maintains the ordered list of messages sent to the model on every request.

    The OpenAI chat completions API is stateless — it has no memory between
    calls. This class provides that memory by accumulating every user and
    assistant message and passing the full list with each new request.

    Args:
        system_prompt: The instruction placed at position 0 in the message list.
            Controls the model's persona and behaviour for the session.
    """

    def __init__(self, system_prompt: str) -> None:
        self._messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]

    # ── mutations ────────────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        """Append a user message to the history.

        Args:
            content: The raw text typed by the user.
        """
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Append an assistant message to the history.

        Args:
            content: The full reply returned by the model.
        """
        self._messages.append({"role": "assistant", "content": content})

    def pop_last_user(self) -> None:
        """Remove the most recent user message.

        Called when a request fails after the user message has already been
        added, so the next attempt does not double-send it. Safe to call
        when the last message is not a user message or the history is empty.
        """
        if len(self._messages) > 1 and self._messages[-1]["role"] == "user":
            self._messages.pop()
            logger.debug("Rolled back last user message after request failure")

    def replace_system_prompt(self, system_prompt: str) -> None:
        """Swap the system prompt without discarding conversation history.

        Args:
            system_prompt: The new instruction to place at position 0.
        """
        self._messages[0] = {"role": "system", "content": system_prompt}
        logger.debug("System prompt updated")

    def clear(self, system_prompt: str) -> None:
        """Reset to a fresh conversation, keeping the provided system prompt.

        Args:
            system_prompt: The instruction to seed the new conversation with.
        """
        self._messages = [{"role": "system", "content": system_prompt}]
        logger.debug("Conversation history reset")

    # ── accessors ────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        """The full message list, ready to pass directly to the completions API.

        Returns:
            Ordered list of messages starting with the system prompt, followed
            by alternating user and assistant turns for the current session.
        """
        return self._messages
