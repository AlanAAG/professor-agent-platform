import unittest
from unittest.mock import patch

from src.shared.utils import cohere_rerank


class TestCohereRerankOrdering(unittest.TestCase):
    def test_returns_expected_order_with_sample_scores_and_embeddings(self):
        # Create sample documents with deterministic IDs and embeddings
        docs = [
            {"id": "d1", "content": "alpha", "metadata": {"embedding": [1.0, 0.0, 0.0]}},
            {"id": "d2", "content": "beta",  "metadata": {"embedding": [0.0, 1.0, 0.0]}},
            {"id": "d3", "content": "gamma", "metadata": {"embedding": [0.0, 0.0, 1.0]}},
        ]

        # Fakes for deterministic, offline behavior
        def fake_expand_query(q: str):
            # Keep a single variant so RRF equals the provided retrieval order
            return [q]

        def fake_retrieve_rag_documents(query: str, selected_class=None, match_count=30, match_threshold=0.7, query_embedding=None):
            # Return a fixed ranking list: d1 > d2 > d3
            return [docs[0], docs[1], docs[2]]

        def fake_embed_query(text: str, model=None):
            # Query embedding aligned to d1 to make MMR preserve the relevance order
            return [1.0, 0.0, 0.0]

        def fake_embed_queries_batch(texts, model=None):
            # Batch embedding for the single expanded query
            return [[1.0, 0.0, 0.0] for _ in texts]

        with patch("src.shared.utils.expand_query", side_effect=fake_expand_query), \
             patch("src.shared.utils.retrieve_rag_documents", side_effect=fake_retrieve_rag_documents), \
             patch("src.shared.utils.embed_query", side_effect=fake_embed_query), \
             patch("src.shared.utils.embed_queries_batch", side_effect=fake_embed_queries_batch):

            result = cohere_rerank("test query", docs)
            result_ids = [d.get("id") or d.get("metadata", {}).get("id") for d in result]

            # Expect the same order enforced by retrieval + RRF with low diversity penalty from MMR
            self.assertEqual(result_ids, ["d1", "d2", "d3"])  # protects ranking order


if __name__ == "__main__":
    unittest.main() 
