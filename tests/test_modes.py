import dataclasses

import pytest

from agent.modes import DEFAULT_MODE, MODES, Mode


def test_default_mode_exists_in_modes():
    assert DEFAULT_MODE in MODES


def test_all_modes_are_mode_instances():
    for mode in MODES.values():
        assert isinstance(mode, Mode)


def test_all_modes_have_non_empty_fields():
    for mode in MODES.values():
        assert mode.name.strip()
        assert mode.description.strip()
        assert mode.system_prompt.strip()


def test_mode_name_matches_dict_key():
    for key, mode in MODES.items():
        assert mode.name == key


def test_modes_are_immutable():
    mode = MODES["CHAT"]
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        mode.name = "hacked"  # type: ignore[misc]


def test_chat_mode_exists():
    assert "CHAT" in MODES


def test_code_mode_exists():
    assert "CODE" in MODES


def test_code_and_chat_have_distinct_system_prompts():
    assert MODES["CHAT"].system_prompt != MODES["CODE"].system_prompt
