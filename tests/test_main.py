"""Unit tests for main.py — setup_logging, complete, and the REPL loop."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError

from agent.conversation import Conversation
from agent.modes import MODES
from main import complete, run, setup_logging

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(content: str | None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices[0].delta.content = content
    return chunk


def _make_stream(*contents: str | None):
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


def test_run_empty_input_is_skipped(mock_run_deps: MagicMock) -> None:
    with patch("builtins.input", side_effect=["", "  ", "/quit"]):
        run()


def test_run_empty_input_does_not_call_complete(mock_run_deps: MagicMock) -> None:
    with (
        patch("builtins.input", side_effect=["", "  ", "/quit"]),
        patch("main.complete") as mock_complete,
    ):
        run()
    mock_complete.assert_not_called()


def test_run_user_message_triggers_complete(mock_run_deps: MagicMock) -> None:
    with (
        patch("builtins.input", side_effect=["hello", "/quit"]),
        patch("main.complete", return_value="hi") as mock_complete,
    ):
        run()
    mock_complete.assert_called_once()
    conv = mock_complete.call_args[0][2]
    user_contents = [m["content"] for m in conv.recent_messages() if m["role"] == "user"]
    assert "hello" in user_contents


def test_run_assistant_reply_is_added_to_history(mock_run_deps: MagicMock) -> None:
    with (
        patch("builtins.input", side_effect=["first", "second", "/quit"]),
        patch("main.complete", return_value="the reply") as mock_complete,
    ):
        run()
    # The second complete() call should see the assistant reply from the first turn
    second_conv = mock_complete.call_args_list[1][0][2]
    assistant_contents = [
        m["content"] for m in second_conv.recent_messages() if m["role"] == "assistant"
    ]
    assert "the reply" in assistant_contents


def test_run_mode_switch_to_code_updates_system_prompt(mock_run_deps: MagicMock) -> None:
    with (
        patch("builtins.input", side_effect=["/code", "hello", "/quit"]),
        patch("main.complete", return_value="response") as mock_complete,
    ):
        run()
    conv = mock_complete.call_args[0][2]
    assert conv.recent_messages()[0]["content"] == MODES["CODE"].system_prompt


def test_run_clear_removes_prior_history(mock_run_deps: MagicMock) -> None:
    with (
        patch("builtins.input", side_effect=["first", "/clear", "second", "/quit"]),
        patch("main.complete", return_value="reply") as mock_complete,
    ):
        run()
    # The second complete() call should not see "first" in the conversation
    second_conv = mock_complete.call_args_list[1][0][2]
    all_contents = [m["content"] for m in second_conv.recent_messages()]
    assert "first" not in all_contents


def test_run_connection_error_rolls_back_and_continues(mock_run_deps: MagicMock) -> None:
    request = httpx.Request("POST", "http://localhost:8083")
    conn_err = APIConnectionError(request=request)
    with (
        patch("builtins.input", side_effect=["hello", "world", "/quit"]),
        patch("main.complete", side_effect=[conn_err, "reply"]) as mock_complete,
    ):
        run()
    # Both user messages should have triggered complete()
    assert mock_complete.call_count == 2
    # After the rollback, "hello" should not appear in the second call's history
    second_conv = mock_complete.call_args_list[1][0][2]
    user_contents = [m["content"] for m in second_conv.recent_messages() if m["role"] == "user"]
    assert "hello" not in user_contents
    assert "world" in user_contents
