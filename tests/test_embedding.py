from unittest.mock import MagicMock
import datetime
import pytest

from src.refinery import embedding


def test_validate_metadata_missing_required_fields():
    with pytest.raises(ValueError):
        embedding.validate_metadata({})
    with pytest.raises(ValueError):
        embedding.validate_metadata({"class_name": "C1"})  # missing content_type
    with pytest.raises(ValueError):
        embedding.validate_metadata({"content_type": "transcript"})  # missing class_name


def test_validate_metadata_success_and_prune_none():
    meta = {
        "class_name": "C1",
        "content_type": "transcript",
        "source_file": None,
        "title": "Intro",
    }
    out = embedding.validate_metadata(meta)
    assert out["class_name"] == "C1"
    assert out["content_type"] == "transcript"
    assert "source_file" not in out  # pruned None
    assert "retrieval_date" in out
    # Should be ISO formatted timestamp string
    datetime.datetime.fromisoformat(out["retrieval_date"])  # no exception


def test_chunk_and_embed_text(monkeypatch):
    # Fake text splitter returns deterministic number of docs
    class Doc:
        def __init__(self):
            self.metadata = {}

    fake_docs = [Doc(), Doc(), Doc()]
    monkeypatch.setattr(embedding.text_splitter, "create_documents", lambda texts: fake_docs)

    # Mock vector_store to observe add_documents call
    fake_vs = MagicMock()
    monkeypatch.setattr(embedding, "vector_store", fake_vs, raising=True)

    metadata = {"class_name": "C1", "content_type": "transcript", "title": "T"}
    embedding.chunk_and_embed_text("some clean text", metadata)

    fake_vs.add_documents.assert_called_once()
    (docs_arg,), _ = fake_vs.add_documents.call_args
    assert len(docs_arg) == len(fake_docs)
    # Ensure metadata was normalized and applied to each doc
    for d in docs_arg:
        assert d.metadata["class_name"] == "C1"
        assert d.metadata["content_type"] == "transcript"
        assert "retrieval_date" in d.metadata


def test_url_exists_in_db_sync_true_and_false(monkeypatch):
    # Build a minimal chainable fake supabase query interface
    class Query:
        def __init__(self, count_value):
            self._count_value = count_value

        def select(self, *_a, **_k):
            return self

        def eq(self, *_a, **_k):
            return self

        def limit(self, *_a, **_k):
            return self

        def execute(self):
            return type("Resp", (), {"count": self._count_value})()

    class FakeSupabase:
        def __init__(self, count_value):
            self._count_value = count_value

        def table(self, *_a, **_k):
            return Query(self._count_value)

    # True case
    monkeypatch.setattr(embedding, "supabase", FakeSupabase(1), raising=True)
    assert embedding.url_exists_in_db_sync("https://example.com/file.pdf") is True

    # False case
    monkeypatch.setattr(embedding, "supabase", FakeSupabase(0), raising=True)
    assert embedding.url_exists_in_db_sync("https://example.com/file.pdf") is False
