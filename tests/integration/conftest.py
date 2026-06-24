"""
Shared fixtures for integration tests.

Integration tests require mimOE Studio to be running with a model loaded.
All fixtures in this file skip automatically when the endpoint is not
configured or not reachable, so they never fail in a cold environment.
"""

from __future__ import annotations

import os

import pytest
from openai import APIConnectionError, OpenAI


@pytest.fixture(scope="session")
def live_client() -> OpenAI:
    """Return an OpenAI client connected to the live mimOE endpoint.

    Returns:
        A configured client pointed at MIMOE_BASE_URL.

    Raises:
        pytest.skip: If MIMOE_BASE_URL is not set or the endpoint is unreachable.
    """
    base_url = os.environ.get("MIMOE_BASE_URL", "").strip()
    if not base_url:
        pytest.skip("MIMOE_BASE_URL not set — start mimOE Studio and configure .env")

    api_key = os.environ.get("MIMOE_API_KEY", "not-required") or "not-required"
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Probe the endpoint before running any tests so all tests in the session
    # are skipped together rather than each failing individually.
    try:
        client.models.list()
    except APIConnectionError:
        pytest.skip(f"mimOE endpoint not reachable at {base_url} — is Studio running?")

    return client


@pytest.fixture(scope="session")
def live_model() -> str:
    """Return the model identifier from environment config.

    Returns:
        The MIMOE_MODEL string (e.g. ``"SmolLM2"``).

    Raises:
        pytest.skip: If MIMOE_MODEL is not set.
    """
    model = os.environ.get("MIMOE_MODEL", "").strip()
    if not model:
        pytest.skip("MIMOE_MODEL not set — configure .env")
    return model
