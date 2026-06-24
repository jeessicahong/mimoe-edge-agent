"""
Builds the OpenAI-compatible client pointed at the local mimOE endpoint.

All connection details come from environment variables so nothing is
hardcoded — copy .env.example to .env and fill in the values from
mimOE Studio's API button.
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)


def build_client() -> OpenAI:
    """Build an OpenAI client pointed at the local mimOE inference endpoint.

    Reads connection details from environment variables. Exits the process
    immediately with a descriptive message if MIMOE_BASE_URL is missing or blank.

    Returns:
        A configured OpenAI client whose requests are directed to the mimOE
        base URL instead of the default OpenAI servers.
    """
    base_url = os.environ.get("MIMOE_BASE_URL", "").strip()
    if not base_url:
        _fatal(
            "MIMOE_BASE_URL is not set.\n"
            "  1. Open mimOE Studio → Model View → API button.\n"
            "  2. Copy the base URL into your .env file."
        )

    # mimOE may not require a real API key; a non-empty placeholder satisfies
    # the OpenAI client's validation without affecting the local endpoint.
    api_key = os.environ.get("MIMOE_API_KEY", "not-required") or "not-required"

    return OpenAI(base_url=base_url, api_key=api_key)


def get_model() -> str:
    """Read the model identifier from environment variables.

    Exits the process immediately with a descriptive message if MIMOE_MODEL
    is missing or blank.

    Returns:
        The model name string to pass as the ``model`` field in every
        chat completions request (e.g. ``"SmolLM2"``).
    """
    model = os.environ.get("MIMOE_MODEL", "").strip()
    if not model:
        _fatal(
            "MIMOE_MODEL is not set.\n"
            "  Set it to the model name shown in mimOE Studio (e.g. SmolLM2)."
        )
    return model


def _fatal(message: str) -> None:
    """Log a critical error and terminate the process.

    Args:
        message: Human-readable description of the configuration problem,
            including remediation steps where possible.
    """
    logger.critical(message)
    sys.exit(1)
