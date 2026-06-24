"""
Interaction modes — each mode is a named system prompt persona.

Switching modes is the lightweight 'agentic' behavior in this agent:
the system prompt changes to route the model's responses through a
different context, while conversation history is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    name: str
    description: str
    system_prompt: str


MODES: dict[str, Mode] = {
    "CHAT": Mode(
        name="CHAT",
        description="General-purpose conversational assistant",
        system_prompt=(
            "You are a helpful, concise assistant. "
            "Answer questions clearly and directly. "
            "If you are uncertain, say so rather than guessing."
        ),
    ),
    "CODE": Mode(
        name="CODE",
        description="Code-focused assistant — writes and explains code",
        system_prompt=(
            "You are an expert programming assistant. "
            "When asked to write code, produce clean, readable, well-structured output. "
            "Briefly explain what the code does and call out any important caveats. "
            "Prefer simple solutions over clever ones."
        ),
    ),
}

DEFAULT_MODE = "CHAT"
