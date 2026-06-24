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

from agent.client import build_client, get_model
from agent.conversation import Conversation
from agent.modes import DEFAULT_MODE, MODES, Mode

logger = logging.getLogger(__name__)

HELP_TEXT = textwrap.dedent("""\
    Commands:
      /chat   — switch to general assistant mode
      /code   — switch to code assistant mode
      /clear  — clear conversation history (keeps current mode)
      /mode   — show current mode
      /help   — show this message
      /quit   — exit
""")


class _TerminalFormatter(logging.Formatter):
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
    handler.setFormatter(_TerminalFormatter())
    logging.basicConfig(level=level, handlers=[handler])

    # httpx logs every HTTP request at INFO level — always suppress it.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Suppress the openai SDK's verbose output in debug mode only.
    if level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.WARNING)


def complete(client: OpenAI, model: str, conversation: Conversation) -> str:
    """Stream the model reply token by token, printing each piece as it arrives.

    Args:
        client: The configured OpenAI client pointed at the mimOE endpoint.
        model: The model identifier to pass in the request (e.g. ``"SmolLM2"``).
        conversation: The current conversation, whose full message history is
            sent with the request.

    Returns:
        The complete assistant reply assembled from all streamed chunks.
    """
    stream = client.chat.completions.create(
        model=model,
        messages=conversation.messages,
        stream=True,
    )
    full_reply = ""
    for chunk in stream:
        piece = chunk.choices[0].delta.content or ""
        print(piece, end="", flush=True)
        full_reply += piece
    print()
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

        print(f"\nAgent ({current_mode.name}): ", end="", flush=True)
        try:
            reply = complete(client, model, conversation)
        except APIConnectionError:
            print()  # close the half-printed prefix line
            conversation.pop_last_user()
            logger.error(
                "Connection failed. Could not reach the mimOE endpoint.\n"
                "  • Is mimOE Studio running and the model loaded?\n"
                "  • Does MIMOE_BASE_URL in .env match the URL in the API button?"
            )
            continue
        except APIStatusError as exc:
            print()  # close the half-printed prefix line
            conversation.pop_last_user()
            logger.error("API error %s: %s", exc.status_code, exc.message)
            continue

        conversation.add_assistant(reply)


if __name__ == "__main__":
    run()
