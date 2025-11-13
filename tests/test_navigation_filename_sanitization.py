import re

from src.harvester.navigation import _sanitize_filename_component


def test_sanitize_filename_component_strips_invalid_characters():
    raw = "find_timeout_xpath_*_self::p or self::div or self::span or self::a"
    sanitized = _sanitize_filename_component(raw)
    assert re.fullmatch(r"[\w.-]+", sanitized)
    assert ":" not in sanitized
    assert "*" not in sanitized


def test_sanitize_filename_component_handles_empty_input():
    assert _sanitize_filename_component("") == "artifact"
