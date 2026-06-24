"""
Terminal display utilities: logging formatter and text post-processing.
"""

from __future__ import annotations

import logging
import re

# Small local models often output LaTeX math regardless of instructions.
# These patterns convert delimiters to markdown equivalents rich can render.
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"\$(.+?)\$")


def strip_latex(text: str) -> str:
    """Replace LaTeX math delimiters with markdown equivalents.

    Display math ($$...$$) becomes a fenced code block so it is visually
    distinct. Inline math ($...$) becomes a backtick span. The LaTeX source
    inside is preserved unchanged — it is not converted to Unicode.

    Args:
        text: Raw model output, potentially containing LaTeX delimiters.

    Returns:
        The same text with LaTeX delimiters replaced by markdown equivalents.
    """
    text = _DISPLAY_MATH.sub(lambda m: f"\n```\n{m.group(1).strip()}\n```\n", text)
    text = _INLINE_MATH.sub(lambda m: f"`{m.group(1).strip()}`", text)
    return text


class TerminalFormatter(logging.Formatter):
    """Custom log formatter for interactive terminal output.

    INFO lines are printed as-is — they are status lines, not log entries.
    DEBUG lines include the logger name so the source is traceable.
    WARNING and above show the level so problems stand out visually.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record for terminal display.

        Args:
            record: The log record produced by a logger call.

        Returns:
            A formatted string. INFO records return the bare message;
            DEBUG records are prefixed with the logger name; all other
            levels are prefixed with the level name.
        """
        if record.levelno == logging.INFO:
            return record.getMessage()
        if record.levelno == logging.DEBUG:
            return f"DEBUG [{record.name}]: {record.getMessage()}"
        return f"{record.levelname}: {record.getMessage()}"
