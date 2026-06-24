"""
In-session conversation history.

Keeps the full message list in memory; history is lost on restart by design
(the assessment scope does not require persistence).
"""

from __future__ import annotations

import logging

from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)


class Conversation:
    def __init__(self, system_prompt: str) -> None:
        self._messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]

    # ── mutations ────────────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def pop_last_user(self) -> None:
        """Remove the most recent user message (used to roll back a failed request)."""
        if len(self._messages) > 1 and self._messages[-1]["role"] == "user":
            self._messages.pop()
            logger.debug("Rolled back last user message after request failure")

    def replace_system_prompt(self, system_prompt: str) -> None:
        """Swap the system prompt without discarding conversation history."""
        self._messages[0] = {"role": "system", "content": system_prompt}
        logger.debug("System prompt updated")

    def clear(self, system_prompt: str) -> None:
        """Reset to a fresh conversation, keeping the provided system prompt."""
        self._messages = [{"role": "system", "content": system_prompt}]
        logger.debug("Conversation history reset")

    # ── accessors ────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        return self._messages
