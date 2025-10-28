import datetime
import math

import pytest

from src.shared import utils


def test_expand_query_basic():
    variants = utils.expand_query("linear regression")
    assert variants[0] == "linear regression"
    assert "linear regression explanation" in variants
    assert len(variants) == len(set(variants))  # deduped


def test_expand_query_empty():
    assert utils.expand_query("") == [""]


def test_get_doc_id_with_id_and_metadata():
    doc1 = {"id": 123, "content": "a"}
    doc2 = {"metadata": {"chunk_id": "c-456"}, "content": "b"}
    assert utils._get_doc_id(doc1) == "123"
    assert utils._get_doc_id(doc2) == "c-456"


def test_get_doc_id_fallback_hash_is_stable():
    doc = {
        "url": "https://example.com/page",
        "title": "Intro",
        "section": "in_class",
        "content": "Body text"
    }
    h1 = utils._get_doc_id(doc)
    h2 = utils._get_doc_id(doc.copy())
    assert h1 == h2
    assert len(h1) == 32  # md5 hex


@pytest.mark.parametrize(
    "date_str, expected",
    [
        ("22/10/2025", datetime.datetime(2025, 10, 22, 0, 0, tzinfo=datetime.timezone.utc)),
        ("October 22, 2025", datetime.datetime(2025, 10, 22, 0, 0, tzinfo=datetime.timezone.utc)),
        ("2025-10-22", datetime.datetime(2025, 10, 22, 0, 0, tzinfo=datetime.timezone.utc)),
        ("22 Oct 2025 | 10:00 AM", datetime.datetime(2025, 10, 22, 0, 0, tzinfo=datetime.timezone.utc)),
    ],
)
def test_parse_general_date_formats(date_str, expected):
    dt = utils.parse_general_date(date_str)
    assert dt is not None
    # Compare date components (ignore minor timezone/local parsing nuances)
    assert dt.year == expected.year and dt.month == expected.month and dt.day == expected.day
    assert dt.tzinfo == datetime.timezone.utc


def test_parse_general_date_invalid():
    assert utils.parse_general_date("not a date at all") is None


def test_maximal_marginal_relevance_relevance_dominates():
    # Three docs: A and B identical embeddings; C orthogonal
    all_docs = {
        "A": {"id": "A", "metadata": {"embedding": [1.0, 0.0]}},
        "B": {"id": "B", "metadata": {"embedding": [1.0, 0.0]}},
        "C": {"id": "C", "metadata": {"embedding": [0.0, 1.0]}},
    }
    scores = {"A": 0.90, "B": 0.85, "C": 0.80}
    q_emb = [1.0, 0.0]

    selected = utils.maximal_marginal_relevance(q_emb, all_docs, scores, lambda_param=0.99, k=2)
    ids = [d["id"] for d in selected]
    # With lambda ~1, order should follow relevance: A then B
    assert ids == ["A", "B"]


def test_maximal_marginal_relevance_diversity_dominates():
    all_docs = {
        "A": {"id": "A", "metadata": {"embedding": [1.0, 0.0]}},
        "B": {"id": "B", "metadata": {"embedding": [1.0, 0.0]}},
        "C": {"id": "C", "metadata": {"embedding": [0.0, 1.0]}},
    }
    scores = {"A": 0.90, "B": 0.85, "C": 0.80}
    q_emb = [1.0, 0.0]

    selected = utils.maximal_marginal_relevance(q_emb, all_docs, scores, lambda_param=0.01, k=2)
    ids = [d["id"] for d in selected]
    # First by relevance (A), then by diversity (C) since B is too similar to A
    assert ids == ["A", "C"]
