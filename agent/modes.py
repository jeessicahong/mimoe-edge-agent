"""
Interaction modes — each mode is a named system prompt persona.

Switching modes is the lightweight 'agentic' behaviour in this agent:
the system prompt changes to route the model's responses through a
different context, while conversation history is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    """An immutable interaction mode defined by a system prompt persona.

    Attributes:
        name: Short uppercase identifier for the mode (e.g. ``"CHAT"``).
        description: One-line human-readable summary shown in the terminal.
        system_prompt: The instruction passed to the model as the first
            message in every conversation. Determines the model's persona
            and response style for the session.
    """

    name: str
    description: str
    system_prompt: str


_FORMAT = (
    "Format responses using markdown. "
    "For mathematical expressions use Unicode symbols "
    "(×, ÷, √, π, ², ³, ≤, ≥, ≠, ∑, ∏) and inline notation rather than LaTeX."
)

MODES: dict[str, Mode] = {
    "CHAT": Mode(
        name="CHAT",
        description="General-purpose conversational assistant",
        system_prompt=(
            "You are a helpful, concise assistant. "
            "Answer questions clearly and directly. "
            f"If you are uncertain, say so rather than guessing. {_FORMAT}"
        ),
    ),
    "CODE": Mode(
        name="CODE",
        description="Code-focused assistant — writes and explains code",
        system_prompt=(
            "You are an expert programming assistant. "
            "When asked to write code, produce clean, readable, well-structured output. "
            "Briefly explain what the code does and call out any important caveats. "
            f"Prefer simple solutions over clever ones. {_FORMAT}"
        ),
    ),
}

DEFAULT_MODE = "CHAT"
