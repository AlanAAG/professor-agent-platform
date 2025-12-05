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
        def __init__(self, content="Chunk content"):
            self.page_content = content
            self.metadata = {}

    fake_docs = [Doc("Chunk 1"), Doc("Chunk 2"), Doc("Chunk 3")]
    monkeypatch.setattr(embedding.text_splitter, "create_documents", lambda texts: fake_docs)

    # Mock get_vector_store to return a fake vector store
    fake_vs = MagicMock()
    monkeypatch.setattr(embedding, "get_vector_store", lambda table_name: fake_vs)

    # Mock _generate_context_summary to return a fixed string
    monkeypatch.setattr(embedding, "_generate_context_summary", lambda text: "This is a mock summary.")

    metadata = {
        "class_name": "C1",
        "content_type": "transcript",
        "title": "T",
        "teacher_name": "Prof. Y",
        "lecture_date": "2023-01-01"
    }
    embedding.chunk_and_embed_text("some clean text", metadata)

    fake_vs.add_documents.assert_called_once()
    (docs_arg,), _ = fake_vs.add_documents.call_args
    assert len(docs_arg) == len(fake_docs)

    # Expected header
    expected_header = (
        "Context Summary: This is a mock summary.\n"
        "Course: C1\n"
        "Source: T\n"
        "Instructor: Prof. Y\n"
        "Date: 2023-01-01\n"
        "---\n"
    )

    # Ensure metadata was normalized and applied to each doc, AND header injected
    for i, d in enumerate(docs_arg):
        assert d.metadata["class_name"] == "C1"
        assert d.metadata["content_type"] == "transcript"
        assert "retrieval_date" in d.metadata

        # Verify content injection
        assert d.page_content.startswith(expected_header)
        assert f"Chunk {i+1}" in d.page_content


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

    # True case: count=1
    monkeypatch.setattr(embedding, "supabase", FakeSupabase(1), raising=True)
    assert embedding.url_exists_in_db_sync("https://example.com/file.pdf") is True

    # False case: count=0
    monkeypatch.setattr(embedding, "supabase", FakeSupabase(0), raising=True)
    assert embedding.url_exists_in_db_sync("https://example.com/file.pdf") is False
