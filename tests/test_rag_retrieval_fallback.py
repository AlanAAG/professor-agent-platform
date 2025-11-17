from tests import test_rag_classification as rag_tests


rag_core = rag_tests.rag_core


def test_retrieval_backoff_uses_relaxed_threshold(monkeypatch):
    calls = []

    def fake_retrieve(query, selected_class, match_count, match_threshold, **kwargs):
        calls.append(
            {
                "count": match_count,
                "threshold": match_threshold,
            }
        )
        if len(calls) == 1:
            return []
        return [{"content": "match"}]

    monkeypatch.setattr(rag_core, "retrieve_rag_documents", fake_retrieve, raising=True)
    monkeypatch.setattr(
        rag_core,
        "retrieve_rag_documents_keyword_fallback",
        lambda *a, **k: [],
        raising=True,
    )

    docs = rag_core._retrieve_documents_with_backoff("question", "Startup")

    assert len(docs) == 1
    assert calls[0]["threshold"] == rag_core.STRICT_MATCH_THRESHOLD
    assert calls[1]["threshold"] == rag_core.RELAXED_MATCH_THRESHOLD
    assert calls[1]["count"] >= rag_core.INITIAL_RETRIEVAL_K


def test_retrieval_backoff_uses_keyword_fallback(monkeypatch):
    monkeypatch.setattr(
        rag_core,
        "retrieve_rag_documents",
        lambda *a, **k: [],
        raising=True,
    )
    captured_limits = []
    fallback_payload = [{"content": "keyword"}]

    def fake_keyword(query, selected_class, limit):
        captured_limits.append(limit)
        return fallback_payload

    monkeypatch.setattr(
        rag_core,
        "retrieve_rag_documents_keyword_fallback",
        fake_keyword,
        raising=True,
    )

    docs = rag_core._retrieve_documents_with_backoff("question", "Startup")

    assert docs == fallback_payload
    assert captured_limits[0] == rag_core.KEYWORD_FALLBACK_LIMIT
