import logging

import pytest

from agent.utils.terminal import TerminalFormatter, strip_latex

# ── TerminalFormatter ─────────────────────────────────────────────────────────


@pytest.fixture
def formatter() -> TerminalFormatter:
    return TerminalFormatter()


def _make_record(level: int, message: str, name: str = "test") -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_info_returns_bare_message(formatter: TerminalFormatter) -> None:
    record = _make_record(logging.INFO, "server started")
    assert formatter.format(record) == "server started"


def test_info_has_no_level_prefix(formatter: TerminalFormatter) -> None:
    record = _make_record(logging.INFO, "ready")
    assert "INFO" not in formatter.format(record)


def test_debug_includes_logger_name(formatter: TerminalFormatter) -> None:
    record = _make_record(logging.DEBUG, "rolling back", name="agent.conversation")
    result = formatter.format(record)
    assert "agent.conversation" in result
    assert "rolling back" in result


def test_debug_includes_debug_prefix(formatter: TerminalFormatter) -> None:
    record = _make_record(logging.DEBUG, "msg")
    assert formatter.format(record).startswith("DEBUG")


def test_warning_includes_level_prefix(formatter: TerminalFormatter) -> None:
    record = _make_record(logging.WARNING, "something suspicious")
    result = formatter.format(record)
    assert result.startswith("WARNING")
    assert "something suspicious" in result


def test_error_includes_level_prefix(formatter: TerminalFormatter) -> None:
    record = _make_record(logging.ERROR, "connection failed")
    result = formatter.format(record)
    assert result.startswith("ERROR")
    assert "connection failed" in result


def test_critical_includes_level_prefix(formatter: TerminalFormatter) -> None:
    record = _make_record(logging.CRITICAL, "fatal config error")
    result = formatter.format(record)
    assert result.startswith("CRITICAL")
    assert "fatal config error" in result


# ── strip_latex ───────────────────────────────────────────────────────────────


def test_plain_text_is_unchanged() -> None:
    assert strip_latex("hello world") == "hello world"


def test_inline_math_becomes_backtick_span() -> None:
    result = strip_latex("The value is $x^2$.")
    assert "`x^2`" in result
    assert "$" not in result


def test_display_math_becomes_code_block() -> None:
    result = strip_latex("Formula: $$a^2 + b^2 = c^2$$")
    assert "```" in result
    assert "a^2 + b^2 = c^2" in result
    assert "$$" not in result


def test_multiple_inline_expressions_are_all_replaced() -> None:
    result = strip_latex("$a$ and $b$")
    assert result.count("`") == 4  # two backtick spans: `a` and `b`
    assert "$" not in result


def test_multiline_display_math_is_handled() -> None:
    text = "$$\na + b\n= c\n$$"
    result = strip_latex(text)
    assert "```" in result
    assert "$$" not in result


def test_display_math_stripped_before_inline() -> None:
    # Ensures $$...$$ is not partially matched by the $...$ pattern
    result = strip_latex("$$x + y$$")
    assert "$$" not in result
    assert "```" in result


def test_text_with_no_math_delimiters_is_unchanged() -> None:
    text = "Use **bold** and `code` in markdown."
    assert strip_latex(text) == text
