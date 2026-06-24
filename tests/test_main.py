"""Unit tests for main.py — setup_logging, complete, and the REPL loop."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from collections.abc import Iterator

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from agent.conversation import Conversation
from agent.modes import MODES
from main import complete, run, setup_logging

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


def _make_stream(*contents: str | None) -> Iterator[MagicMock]:
    return iter(_make_chunk(c) for c in contents)


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_loggers():
    """Clear root logger handlers before each test so basicConfig() runs fresh."""
    root = logging.getLogger()
    orig_handlers = list(root.handlers)
    orig_level = root.level
    root.handlers.clear()
    yield
    root.handlers.clear()
    for h in orig_handlers:
        root.addHandler(h)
    root.setLevel(orig_level)
    logging.getLogger("httpx").setLevel(logging.NOTSET)
    logging.getLogger("openai").setLevel(logging.NOTSET)


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.base_url = "http://localhost:8083"
    return client


@pytest.fixture
def conversation() -> Conversation:
    c = Conversation("you are helpful")
    c.add_user("hello")
    return c


@pytest.fixture
def mock_run_deps(mock_client: MagicMock):
    """Patch the external dependencies of run() so tests control the REPL."""
    with (
        patch("main.build_client", return_value=mock_client),
        patch("main.get_model", return_value="test-model"),
        patch("main.setup_logging"),
    ):
        yield mock_client


# ── setup_logging ─────────────────────────────────────────────────────────────


def test_setup_logging_default_level_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch basicConfig to check the level argument directly — pytest's logging
    # plugin adds handlers before basicConfig runs, which would make it a no-op
    # and prevent the root logger level from being set by the real call.
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    with patch("logging.basicConfig") as mock_basicconfig:
        setup_logging()
    assert mock_basicconfig.call_args.kwargs["level"] == logging.INFO


def test_setup_logging_reads_log_level_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    with patch("logging.basicConfig") as mock_basicconfig:
        setup_logging()
    assert mock_basicconfig.call_args.kwargs["level"] == logging.DEBUG


def test_setup_logging_invalid_level_falls_back_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "NOTVALID")
    with patch("logging.basicConfig") as mock_basicconfig:
        setup_logging()
    assert mock_basicconfig.call_args.kwargs["level"] == logging.INFO


def test_setup_logging_always_suppresses_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    setup_logging()
    assert logging.getLogger("httpx").level == logging.WARNING


def test_setup_logging_suppresses_openai_in_debug_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    setup_logging()
    assert logging.getLogger("openai").level == logging.WARNING


def test_setup_logging_does_not_suppress_openai_in_info_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    setup_logging()
    assert logging.getLogger("openai").level == logging.NOTSET


# ── complete ──────────────────────────────────────────────────────────────────


def test_complete_assembles_chunks_into_full_reply(
    mock_client: MagicMock, conversation: Conversation
) -> None:
    mock_client.chat.completions.create.return_value = _make_stream("Hello", ", ", "world!")
    with patch("main.Live"):
        result = complete(mock_client, "test-model", conversation)
    assert result == "Hello, world!"


def test_complete_handles_none_content_in_chunks(
    mock_client: MagicMock, conversation: Conversation
) -> None:
    mock_client.chat.completions.create.return_value = _make_stream("Hi", None, "!")
    with patch("main.Live"):
        result = complete(mock_client, "test-model", conversation)
    assert result == "Hi!"


def test_complete_passes_correct_model_to_client(
    mock_client: MagicMock, conversation: Conversation
) -> None:
    mock_client.chat.completions.create.return_value = _make_stream("ok")
    with patch("main.Live"):
        complete(mock_client, "my-model", conversation)
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "my-model"


def test_complete_sends_conversation_messages(
    mock_client: MagicMock, conversation: Conversation
) -> None:
    mock_client.chat.completions.create.return_value = _make_stream("ok")
    with patch("main.Live"):
        complete(mock_client, "test-model", conversation)
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == conversation.recent_messages()


def test_complete_returns_empty_string_for_empty_stream(
    mock_client: MagicMock, conversation: Conversation
) -> None:
    mock_client.chat.completions.create.return_value = iter([])
    with patch("main.Live"):
        result = complete(mock_client, "test-model", conversation)
    assert result == ""


# ── run ───────────────────────────────────────────────────────────────────────


def test_run_quit_exits_cleanly(mock_run_deps: MagicMock) -> None:
    with patch("builtins.input", side_effect=["/quit"]):
        run()


def test_run_eof_exits_cleanly(mock_run_deps: MagicMock) -> None:
    with patch("builtins.input", side_effect=EOFError):
        run()


def test_run_keyboard_interrupt_exits_cleanly(mock_run_deps: MagicMock) -> None:
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        run()


def test_run_empty_input_does_not_call_complete(mock_run_deps: MagicMock) -> None:
    with patch("builtins.input", side_effect=["", "  ", "/quit"]):
        run()
    mock_run_deps.chat.completions.create.assert_not_called()


def test_run_user_message_triggers_complete(mock_run_deps: MagicMock) -> None:
    mock_run_deps.chat.completions.create.side_effect = lambda **_: _make_stream("hi")
    with (
        patch("builtins.input", side_effect=["hello", "/quit"]),
        patch("main.Live"),
    ):
        run()
    messages = mock_run_deps.chat.completions.create.call_args.kwargs["messages"]
    assert any(m["role"] == "user" and m["content"] == "hello" for m in messages)


def test_run_assistant_reply_is_added_to_history(mock_run_deps: MagicMock) -> None:
    mock_run_deps.chat.completions.create.side_effect = lambda **_: _make_stream("the reply")
    with (
        patch("builtins.input", side_effect=["first", "second", "/quit"]),
        patch("main.Live"),
    ):
        run()
    second_messages = mock_run_deps.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert any(m["role"] == "assistant" and m["content"] == "the reply" for m in second_messages)


def test_run_help_command_prints_help_text(capsys: pytest.CaptureFixture) -> None:
    with (
        patch("main.build_client"),
        patch("main.get_model", return_value="test-model"),
        patch("main.setup_logging"),
        patch("builtins.input", side_effect=["/help", "/quit"]),
    ):
        run()
    assert "/chat" in capsys.readouterr().out


def test_run_mode_command_logs_current_mode(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch("main.build_client"),
        patch("main.get_model", return_value="test-model"),
        patch("main.setup_logging"),
        patch("builtins.input", side_effect=["/mode", "/quit"]),
        caplog.at_level(logging.INFO),
    ):
        run()
    assert "CHAT" in caplog.text


def test_run_unknown_command_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    with (
        patch("main.build_client"),
        patch("main.get_model", return_value="test-model"),
        patch("main.setup_logging"),
        patch("builtins.input", side_effect=["/foo", "/quit"]),
        caplog.at_level(logging.WARNING),
    ):
        run()
    assert "/foo" in caplog.text


def test_run_mode_switch_to_code_updates_system_prompt(mock_run_deps: MagicMock) -> None:
    mock_run_deps.chat.completions.create.side_effect = lambda **_: _make_stream("response")
    with (
        patch("builtins.input", side_effect=["/code", "hello", "/quit"]),
        patch("main.Live"),
    ):
        run()
    messages = mock_run_deps.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"] == MODES["CODE"].system_prompt


def test_run_clear_removes_prior_history(mock_run_deps: MagicMock) -> None:
    mock_run_deps.chat.completions.create.side_effect = lambda **_: _make_stream("reply")
    with (
        patch("builtins.input", side_effect=["first", "/clear", "second", "/quit"]),
        patch("main.Live"),
    ):
        run()
    second_messages = mock_run_deps.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert "first" not in [m["content"] for m in second_messages]


def test_run_api_status_error_rolls_back_and_continues(mock_run_deps: MagicMock) -> None:
    request = httpx.Request("POST", "http://localhost:8083")
    mock_run_deps.chat.completions.create.side_effect = [
        APIStatusError("model not found", response=httpx.Response(404, request=request), body=None),
        _make_stream("reply"),
    ]
    with (
        patch("builtins.input", side_effect=["hello", "world", "/quit"]),
        patch("main.Live"),
    ):
        run()
    second_messages = mock_run_deps.chat.completions.create.call_args_list[1].kwargs["messages"]
    user_contents = [m["content"] for m in second_messages if m["role"] == "user"]
    assert "hello" not in user_contents
    assert "world" in user_contents


def test_run_empty_reply_is_not_added_to_history(
    mock_run_deps: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    mock_run_deps.chat.completions.create.side_effect = [
        iter([]),  # empty stream → complete() returns "" → ValueError → warning
        _make_stream("reply"),
    ]
    with (
        patch("builtins.input", side_effect=["hello", "world", "/quit"]),
        patch("main.Live"),
        caplog.at_level(logging.WARNING),
    ):
        run()
    assert "empty" in caplog.text.lower()
    second_messages = mock_run_deps.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert "" not in [m["content"] for m in second_messages if m["role"] == "assistant"]


def test_run_connection_error_rolls_back_and_continues(mock_run_deps: MagicMock) -> None:
    request = httpx.Request("POST", "http://localhost:8083")
    mock_run_deps.chat.completions.create.side_effect = [
        APIConnectionError(request=request),
        _make_stream("reply"),
    ]
    with (
        patch("builtins.input", side_effect=["hello", "world", "/quit"]),
        patch("main.Live"),
    ):
        run()
    second_messages = mock_run_deps.chat.completions.create.call_args_list[1].kwargs["messages"]
    user_contents = [m["content"] for m in second_messages if m["role"] == "user"]
    assert "hello" not in user_contents
    assert "world" in user_contents
