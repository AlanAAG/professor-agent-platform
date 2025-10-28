import pytest
from unittest.mock import MagicMock

from src.refinery import cleaning


def test_clean_transcript_locally_basic():
    raw = (
        "0:01 hello everyone\n"
        "this is a sample line without punctuation\n\n"
        "1:02:03 another line with timestamp\n"
        "small\nfragment\n"
        "final sentence here"
    )

    cleaned = cleaning._clean_transcript_locally(raw)

    assert "0:01" not in cleaned
    assert "1:02:03" not in cleaned
    # Expect sentences end with punctuation
    assert cleaned.strip().endswith(".")
    # Should not contain multiple blank lines in a row
    assert "\n\n\n" not in cleaned


class _FakeChain:
    def __init__(self, output, raise_on_invoke=False):
        self._output = output
        self._raise = raise_on_invoke

    def __or__(self, other):
        return self

    def invoke(self, _):
        if self._raise:
            raise RuntimeError("LLM error")
        return self._output


class _FakePrompt:
    def __init__(self, output, raise_on_invoke=False):
        self._output = output
        self._raise = raise_on_invoke

    def __or__(self, other):
        return _FakeChain(self._output, self._raise)


def test_clean_transcript_with_llm_success(monkeypatch):
    # Build a fake chain via overridden prompt and parser
    monkeypatch.setattr(cleaning, "ChatPromptTemplate", MagicMock())
    monkeypatch.setattr(cleaning.ChatPromptTemplate, "from_template", lambda *_: _FakePrompt("Cleaned by LLM."))
    monkeypatch.setattr(cleaning, "StrOutputParser", lambda *_: object())
    # Ensure model exists (any object that participates in | is fine)
    monkeypatch.setattr(cleaning, "model", object(), raising=True)

    result = cleaning.clean_transcript_with_llm("some raw transcript that is definitely more than fifty characters long to pass the check")
    assert result == "Cleaned by LLM."


def test_clean_transcript_with_llm_fallback(monkeypatch):
    # Force the chain to raise so we exercise the local fallback path
    monkeypatch.setattr(cleaning, "ChatPromptTemplate", MagicMock())
    monkeypatch.setattr(cleaning.ChatPromptTemplate, "from_template", lambda *_: _FakePrompt("", raise_on_invoke=True))
    monkeypatch.setattr(cleaning, "StrOutputParser", lambda *_: object())
    monkeypatch.setattr(cleaning, "model", object(), raising=True)

    raw = "0:01 This should fall back to local cleaning due to error."
    result = cleaning.clean_transcript_with_llm(raw)

    # Local cleaner should have removed the timestamp
    assert "0:01" not in result
