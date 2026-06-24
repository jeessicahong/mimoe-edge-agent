"""
Minimal conversational agent backed by the local mimOE inference endpoint.

Usage:
    python main.py

Special commands (type at the prompt):
    /chat    Switch to general-assistant mode
    /code    Switch to code-assistant mode
    /clear   Clear conversation history (keeps current mode)
    /mode    Show the current mode
    /help    Show available commands
    /quit    Exit
"""

from __future__ import annotations

import logging
import os
import textwrap

from openai import APIConnectionError, APIStatusError, OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from agent.client import build_client, get_model
from agent.conversation import Conversation
from agent.modes import DEFAULT_MODE, MODES, Mode
from agent.utils.terminal import TerminalFormatter, strip_latex

logger = logging.getLogger(__name__)
_console = Console()


HELP_TEXT = textwrap.dedent("""\
    Commands:
      /chat   — switch to general assistant mode
      /code   — switch to code assistant mode
      /clear  — clear conversation history (keeps current mode)
      /mode   — show current mode
      /help   — show this message
      /quit   — exit
""")


def setup_logging() -> None:
    """Configure the root logger for terminal output.

    Log level is controlled by the LOG_LEVEL environment variable (default:
    INFO). Set LOG_LEVEL=DEBUG to see internal state traces from the agent
    modules without HTTP library noise.

    Conversation output (agent replies and the You: prompt) uses print() to
    stdout so it can be piped independently of these diagnostic messages,
    which go to stderr.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(TerminalFormatter())
    logging.basicConfig(level=level, handlers=[handler])

    # httpx logs every HTTP request at INFO level — always suppress it.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Suppress the openai SDK's verbose output in debug mode only.
    if level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.WARNING)


def complete(client: OpenAI, model: str, conversation: Conversation) -> str:
    """Stream the model reply and render it as live markdown in the terminal.

    Tokens are appended to a buffer as they arrive; the buffer is re-rendered
    as markdown on each update so formatting (bold, code blocks, lists) appears
    correctly rather than as raw markup.

    Args:
        client: The configured OpenAI client pointed at the mimOE endpoint.
        model: The model identifier to pass in the request (e.g. ``"SmolLM2"``).
        conversation: The current conversation, whose full message history is
            sent with the request.

    Returns:
        The complete assistant reply assembled from all streamed chunks.
    """
    messages = conversation.recent_messages()
    logger.debug("Sending %d messages to model", len(messages))
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=512,
    )
    full_reply = ""
    with Live(console=_console, refresh_per_second=15) as live:
        for chunk in stream:
            piece = chunk.choices[0].delta.content or ""
            full_reply += piece
            live.update(Markdown(strip_latex(full_reply)))
    return full_reply


def run() -> None:
    """Entry point for the conversational REPL loop.

    Initialises logging, builds the OpenAI client from environment config,
    and drops into an interactive loop that reads user input, dispatches slash
    commands, and streams model replies token by token until the user quits.
    """
    setup_logging()

    client = build_client()
    model = get_model()

    current_mode: Mode = MODES[DEFAULT_MODE]
    conversation = Conversation(current_mode.system_prompt)

    logger.info("mimoe-edge-agent | endpoint: %s | model: %s", client.base_url, model)
    logger.info("mode: %s — %s", current_mode.name, current_mode.description)
    logger.info("Type /help for commands, /quit to exit.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        # ── slash commands ────────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd == "/quit":
                print("Bye.")
                break

            elif cmd == "/help":
                print(HELP_TEXT)

            elif cmd == "/mode":
                logger.info("%s — %s", current_mode.name, current_mode.description)

            elif cmd == "/clear":
                conversation.clear(current_mode.system_prompt)
                logger.info("Conversation cleared. (%s mode retained)", current_mode.name)

            elif cmd in ("/chat", "/code"):
                mode_name = cmd.lstrip("/").upper()
                current_mode = MODES[mode_name]
                conversation.replace_system_prompt(current_mode.system_prompt)
                logger.info(
                    "Switched to %s mode — %s",
                    current_mode.name,
                    current_mode.description,
                )

            else:
                logger.warning(
                    "Unknown command: %s. Type /help for available commands.", user_input
                )

            continue

        # ── inference ─────────────────────────────────────────────────────────
        conversation.add_user(user_input)

        print(f"\nAgent ({current_mode.name}):")
        try:
            reply = complete(client, model, conversation)
        except APIConnectionError:
            conversation.pop_last_user()
            logger.error(
                "Connection failed. Could not reach the mimOE endpoint.\n"
                "  • Is mimOE Studio running and the model loaded?\n"
                "  • Does MIMOE_BASE_URL in .env match the URL in the API button?"
            )
            continue
        except APIStatusError as exc:
            conversation.pop_last_user()
            logger.error("API error %s: %s", exc.status_code, exc.message)
            continue

        conversation.add_assistant(reply)


if __name__ == "__main__":
    run()
