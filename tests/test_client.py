import pytest

from agent.client import build_client, get_model


def test_build_client_exits_when_base_url_is_missing(monkeypatch):
    monkeypatch.delenv("MIMOE_BASE_URL", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        build_client()
    assert exc_info.value.code == 1


def test_build_client_exits_when_base_url_is_blank(monkeypatch):
    monkeypatch.setenv("MIMOE_BASE_URL", "   ")  # whitespace only
    with pytest.raises(SystemExit) as exc_info:
        build_client()
    assert exc_info.value.code == 1


def test_build_client_returns_client_with_valid_url(monkeypatch):
    monkeypatch.setenv("MIMOE_BASE_URL", "http://localhost:8080/v1")
    client = build_client()
    assert client is not None


def test_build_client_uses_placeholder_when_api_key_missing(monkeypatch):
    monkeypatch.setenv("MIMOE_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.delenv("MIMOE_API_KEY", raising=False)
    # should not raise — the placeholder satisfies the OpenAI client
    client = build_client()
    assert client is not None


def test_get_model_exits_when_model_is_missing(monkeypatch):
    monkeypatch.delenv("MIMOE_MODEL", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        get_model()
    assert exc_info.value.code == 1


def test_get_model_exits_when_model_is_blank(monkeypatch):
    monkeypatch.setenv("MIMOE_MODEL", "   ")
    with pytest.raises(SystemExit) as exc_info:
        get_model()
    assert exc_info.value.code == 1


def test_get_model_returns_model_name(monkeypatch):
    monkeypatch.setenv("MIMOE_MODEL", "SmolLM2")
    assert get_model() == "SmolLM2"


def test_get_model_strips_whitespace(monkeypatch):
    monkeypatch.setenv("MIMOE_MODEL", "  SmolLM2  ")
    assert get_model() == "SmolLM2"
