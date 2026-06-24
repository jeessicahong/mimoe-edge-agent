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
    model = os.environ.get("MIMOE_MODEL", "").strip()
    if not model:
        _fatal(
            "MIMOE_MODEL is not set.\n"
            "  Set it to the model name shown in mimOE Studio (e.g. SmolLM2)."
        )
    return model


def _fatal(message: str) -> None:
    logger.critical(message)
    sys.exit(1)
