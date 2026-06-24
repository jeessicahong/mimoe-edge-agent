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

        Raises:
            ValueError: If content is empty or whitespace-only.
        """
        if not content.strip():
            raise ValueError("User message content must not be empty.")
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Append an assistant message to the history.

        Args:
            content: The full reply returned by the model.

        Raises:
            ValueError: If content is empty or whitespace-only.
        """
        if not content.strip():
            raise ValueError("Assistant message content must not be empty.")
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

    def recent_messages(self, max_turns: int = 10) -> list[ChatCompletionMessageParam]:
        """Return the system prompt plus the most recent conversation turns.

        Older turns are dropped once the history exceeds ``max_turns`` pairs,
        preventing context window overflow on models with limited token budgets.
        The system prompt is always included regardless of ``max_turns``.

        Args:
            max_turns: Maximum number of user/assistant pairs to include.
                Each pair counts as two messages. Defaults to 10.

        Returns:
            Ordered list starting with the system prompt, followed by up to
            ``max_turns * 2`` of the most recent user and assistant messages.
        """
        system = self._messages[:1]
        history = self._messages[1:]
        return system + history[-(max_turns * 2) :]
