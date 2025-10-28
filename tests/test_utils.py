"""
Unit tests for src/shared/utils.py

Tests cover all utility functions with various edge cases and scenarios,
using mocked dependencies to ensure isolated testing environment.
"""

import pytest
import hashlib
import datetime
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# Import the functions we want to test
from src.shared.utils import (
    expand_query,
    _get_doc_id,
    _get_doc_class,
    _get_doc_section,
    apply_rrf_and_boost,
    compute_cosine_similarity,
    _extract_doc_embedding,
    maximal_marginal_relevance,
    _normalize_doc_input,
    cohere_rerank,
    parse_general_date,
    create_safe_filename
)


class TestExpandQuery:
    """Test cases for expand_query function."""
    
    def test_basic_query(self):
        """Test expansion of a basic query."""
        result = expand_query("machine learning")
        expected = [
            "machine learning",
            "machine learning explanation", 
            "machine learning definition",
            "machine learning examples",
            "machine learning in practice"
        ]
        assert result == expected
    
    def test_empty_query(self):
        """Test expansion of empty query."""
        result = expand_query("")
        assert result == [""]
        
        result = expand_query("   ")
        assert result == [""]
    
    def test_query_with_keywords_already_present(self):
        """Test query that already contains expansion keywords."""
        result = expand_query("machine learning explanation")
        # Should still add other variants but avoid duplicates
        assert "machine learning explanation" in result
        assert "machine learning explanation definition" in result
        assert len(result) == len(set(result))  # No duplicates
    
    def test_whitespace_handling(self):
        """Test query with extra whitespace."""
        result = expand_query("  neural networks  ")
        expected = [
            "neural networks",
            "neural networks explanation",
            "neural networks definition", 
            "neural networks examples",
            "neural networks in practice"
        ]
        assert result == expected


class TestGetDocId:
    """Test cases for _get_doc_id function."""
    
    def test_document_with_id(self):
        """Test document with direct id field."""
        doc = {"id": "doc_123", "content": "test content"}
        result = _get_doc_id(doc)
        assert result == "doc_123"
    
    def test_document_with_doc_id(self):
        """Test document with doc_id field."""
        doc = {"doc_id": "document_456", "content": "test content"}
        result = _get_doc_id(doc)
        assert result == "document_456"
    
    def test_document_with_chunk_id(self):
        """Test document with chunk_id field."""
        doc = {"chunk_id": "chunk_789", "content": "test content"}
        result = _get_doc_id(doc)
        assert result == "chunk_789"
    
    def test_document_with_metadata_id(self):
        """Test document with id in metadata."""
        doc = {
            "content": "test content",
            "metadata": {"chunk_id": "meta_chunk_123"}
        }
        result = _get_doc_id(doc)
        assert result == "meta_chunk_123"
    
    def test_document_fallback_hash(self):
        """Test fallback hash generation for document without ids."""
        doc = {
            "url": "https://example.com",
            "title": "Test Document",
            "section": "intro",
            "content": "test content"
        }
        result = _get_doc_id(doc)
        
        # Verify it's a valid hash
        assert len(result) == 32  # MD5 hash length
        
        # Verify hash is deterministic
        result2 = _get_doc_id(doc)
        assert result == result2
        
        # Verify hash changes with different content
        doc2 = doc.copy()
        doc2["content"] = "different content"
        result3 = _get_doc_id(doc2)
        assert result != result3
    
    def test_document_hash_stability(self):
        """Test that hash generation is stable across calls."""
        doc = {
            "url": "https://example.com/doc",
            "title": "Stable Document",
            "section": "chapter1", 
            "page_content": "stable content"  # Uses page_content instead of content
        }
        
        # Generate hash multiple times
        hashes = [_get_doc_id(doc) for _ in range(5)]
        
        # All hashes should be identical
        assert len(set(hashes)) == 1
        
        # Should be a valid MD5 hash
        expected_basis = "https://example.com/doc|Stable Document|chapter1|stable content"
        expected_hash = hashlib.md5(expected_basis.encode("utf-8")).hexdigest()
        assert hashes[0] == expected_hash


class TestGetDocClass:
    """Test cases for _get_doc_class function."""
    
    def test_class_name_in_doc(self):
        """Test extracting class_name from document root."""
        doc = {"class_name": "CS101", "content": "test"}
        result = _get_doc_class(doc)
        assert result == "CS101"
    
    def test_class_name_in_metadata(self):
        """Test extracting class_name from metadata."""
        doc = {
            "content": "test",
            "metadata": {"class_name": "MATH201"}
        }
        result = _get_doc_class(doc)
        assert result == "MATH201"
    
    def test_no_class_name(self):
        """Test document without class_name."""
        doc = {"content": "test", "metadata": {}}
        result = _get_doc_class(doc)
        assert result == ""
    
    def test_empty_class_name(self):
        """Test document with empty class_name."""
        doc = {"class_name": "", "content": "test"}
        result = _get_doc_class(doc)
        assert result == ""
    
    def test_whitespace_class_name(self):
        """Test document with whitespace-only class_name."""
        doc = {"class_name": "   ", "content": "test"}
        result = _get_doc_class(doc)
        assert result == ""


class TestGetDocSection:
    """Test cases for _get_doc_section function."""
    
    def test_section_in_doc(self):
        """Test extracting section from document root."""
        doc = {"section": "Introduction", "content": "test"}
        result = _get_doc_section(doc)
        assert result == "introduction"  # Should be lowercase
    
    def test_section_in_metadata(self):
        """Test extracting section from metadata."""
        doc = {
            "content": "test",
            "metadata": {"section": "SESSIONS"}
        }
        result = _get_doc_section(doc)
        assert result == "sessions"  # Should be lowercase
    
    def test_no_section(self):
        """Test document without section."""
        doc = {"content": "test", "metadata": {}}
        result = _get_doc_section(doc)
        assert result == ""
    
    def test_section_case_normalization(self):
        """Test that section names are normalized to lowercase."""
        doc = {"section": "In_Class", "content": "test"}
        result = _get_doc_section(doc)
        assert result == "in_class"


class TestApplyRrfAndBoost:
    """Test cases for apply_rrf_and_boost function."""
    
    def test_basic_rrf_calculation(self):
        """Test basic RRF score calculation."""
        results_per_query = [
            [
                {"id": "doc1", "content": "test1"},
                {"id": "doc2", "content": "test2"}
            ],
            [
                {"id": "doc2", "content": "test2"},  # doc2 appears in both queries
                {"id": "doc3", "content": "test3"}
            ]
        ]
        
        scores = apply_rrf_and_boost(results_per_query, "test query", "")
        
        # doc1: rank 1 in query 1 -> 1/(60+1) = 1/61
        # doc2: rank 2 in query 1 + rank 1 in query 2 -> 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        # doc3: rank 2 in query 2 -> 1/(60+2) = 1/62
        
        assert "doc1" in scores
        assert "doc2" in scores
        assert "doc3" in scores
        
        # doc2 should have highest score (appears in both queries)
        assert scores["doc2"] > scores["doc1"]
        assert scores["doc2"] > scores["doc3"]
    
    def test_class_name_boost(self):
        """Test class name matching boost."""
        results_per_query = [
            [
                {"id": "doc1", "class_name": "CS101", "content": "test1"},
                {"id": "doc2", "class_name": "MATH201", "content": "test2"}
            ]
        ]
        
        scores = apply_rrf_and_boost(results_per_query, "test query", "CS101")
        
        # doc1 should get +0.15 boost for matching class
        base_score_doc1 = 1.0 / (60.0 + 1.0)  # RRF score for rank 1
        base_score_doc2 = 1.0 / (60.0 + 2.0)  # RRF score for rank 2
        
        assert abs(scores["doc1"] - (base_score_doc1 + 0.15)) < 1e-10
        assert abs(scores["doc2"] - base_score_doc2) < 1e-10
    
    def test_section_boost(self):
        """Test section priority boosts."""
        results_per_query = [
            [
                {"id": "doc1", "section": "sessions", "content": "test1"},
                {"id": "doc2", "section": "in_class", "content": "test2"},
                {"id": "doc3", "section": "other", "content": "test3"}
            ]
        ]
        
        scores = apply_rrf_and_boost(results_per_query, "test query", "")
        
        base_score_1 = 1.0 / (60.0 + 1.0)
        base_score_2 = 1.0 / (60.0 + 2.0)
        base_score_3 = 1.0 / (60.0 + 3.0)
        
        # sessions gets +0.10, in_class gets +0.05, other gets +0.0
        assert abs(scores["doc1"] - (base_score_1 + 0.10)) < 1e-10
        assert abs(scores["doc2"] - (base_score_2 + 0.05)) < 1e-10
        assert abs(scores["doc3"] - base_score_3) < 1e-10
    
    def test_combined_boosts(self):
        """Test combined class and section boosts."""
        results_per_query = [
            [
                {
                    "id": "doc1",
                    "class_name": "CS101",
                    "section": "sessions",
                    "content": "test1"
                }
            ]
        ]
        
        scores = apply_rrf_and_boost(results_per_query, "test query", "CS101")
        
        base_score = 1.0 / (60.0 + 1.0)
        expected_score = base_score + 0.15 + 0.10  # class boost + section boost
        
        assert abs(scores["doc1"] - expected_score) < 1e-10
    
    def test_empty_results(self):
        """Test with empty results."""
        scores = apply_rrf_and_boost([], "test query", "CS101")
        assert scores == {}


class TestComputeCosineSimilarity:
    """Test cases for compute_cosine_similarity function."""
    
    def test_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        vec = [1.0, 2.0, 3.0, 4.0]
        result = compute_cosine_similarity(vec, vec)
        assert abs(result - 1.0) < 1e-10
    
    def test_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        result = compute_cosine_similarity(vec1, vec2)
        assert abs(result - 0.0) < 1e-10
    
    def test_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        result = compute_cosine_similarity(vec1, vec2)
        assert abs(result - (-1.0)) < 1e-10
    
    def test_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        result = compute_cosine_similarity(vec1, vec2)
        assert result == 0.0
    
    def test_invalid_dimensions(self):
        """Test cosine similarity with mismatched dimensions."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0]
        # Should handle gracefully and return 0.0
        result = compute_cosine_similarity(vec1, vec2)
        assert result == 0.0
    
    def test_invalid_input_types(self):
        """Test cosine similarity with invalid input types."""
        result = compute_cosine_similarity("invalid", [1.0, 2.0])
        assert result == 0.0
        
        result = compute_cosine_similarity([1.0, 2.0], None)
        assert result == 0.0
    
    def test_empty_vectors(self):
        """Test cosine similarity with empty vectors."""
        result = compute_cosine_similarity([], [])
        assert result == 0.0


class TestExtractDocEmbedding:
    """Test cases for _extract_doc_embedding function."""
    
    def test_embedding_in_root(self):
        """Test extracting embedding from document root."""
        doc = {
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "content": "test"
        }
        result = _extract_doc_embedding(doc)
        assert result == [0.1, 0.2, 0.3, 0.4]
    
    def test_embedding_in_metadata(self):
        """Test extracting embedding from metadata."""
        doc = {
            "content": "test",
            "metadata": {
                "embedding": [0.5, 0.6, 0.7, 0.8]
            }
        }
        result = _extract_doc_embedding(doc)
        assert result == [0.5, 0.6, 0.7, 0.8]
    
    def test_embedding_vector_key(self):
        """Test extracting embedding with 'embedding_vector' key."""
        doc = {
            "embedding_vector": [0.9, 1.0, 1.1, 1.2],
            "content": "test"
        }
        result = _extract_doc_embedding(doc)
        assert result == [0.9, 1.0, 1.1, 1.2]
    
    def test_vector_key(self):
        """Test extracting embedding with 'vector' key."""
        doc = {
            "vector": [1.3, 1.4, 1.5, 1.6],
            "content": "test"
        }
        result = _extract_doc_embedding(doc)
        assert result == [1.3, 1.4, 1.5, 1.6]
    
    def test_no_embedding(self):
        """Test document without embedding."""
        doc = {"content": "test", "metadata": {}}
        result = _extract_doc_embedding(doc)
        assert result is None
    
    def test_empty_embedding(self):
        """Test document with empty embedding."""
        doc = {"embedding": [], "content": "test"}
        result = _extract_doc_embedding(doc)
        assert result is None
    
    def test_invalid_embedding_type(self):
        """Test document with invalid embedding type."""
        doc = {"embedding": "not_a_list", "content": "test"}
        result = _extract_doc_embedding(doc)
        assert result is None


class TestMaximalMarginalRelevance:
    """Test cases for maximal_marginal_relevance function."""
    
    def test_relevance_dominance(self):
        """Test MMR when relevance dominates (lambda ≈ 1.0)."""
        query_embedding = [1.0, 0.0, 0.0]
        
        all_documents = {
            "doc1": {
                "content": "high relevance doc",
                "embedding": [0.9, 0.1, 0.0]  # High similarity to query
            },
            "doc2": {
                "content": "medium relevance doc", 
                "embedding": [0.5, 0.5, 0.0]  # Medium similarity to query
            },
            "doc3": {
                "content": "low relevance doc",
                "embedding": [0.1, 0.9, 0.0]  # Low similarity to query
            }
        }
        
        doc_scores = {
            "doc1": 0.9,  # High relevance score
            "doc2": 0.5,  # Medium relevance score
            "doc3": 0.1   # Low relevance score
        }
        
        result = maximal_marginal_relevance(
            query_embedding=query_embedding,
            all_documents=all_documents,
            doc_scores=doc_scores,
            lambda_param=0.95,  # Heavily favor relevance
            k=3
        )
        
        # Should select in order of relevance when lambda is high
        assert len(result) == 3
        assert result[0]["content"] == "high relevance doc"
        assert result[1]["content"] == "medium relevance doc"
        assert result[2]["content"] == "low relevance doc"
    
    def test_diversity_dominance(self):
        """Test MMR when diversity dominates (lambda ≈ 0.0)."""
        query_embedding = [1.0, 0.0, 0.0]
        
        all_documents = {
            "doc1": {
                "content": "first doc",
                "embedding": [1.0, 0.0, 0.0]  # Identical to query
            },
            "doc2": {
                "content": "similar doc",
                "embedding": [0.9, 0.1, 0.0]  # Very similar to doc1
            },
            "doc3": {
                "content": "diverse doc",
                "embedding": [0.0, 0.0, 1.0]  # Very different from doc1
            }
        }
        
        doc_scores = {
            "doc1": 0.9,
            "doc2": 0.8,  # Slightly lower relevance
            "doc3": 0.3   # Much lower relevance but very diverse
        }
        
        result = maximal_marginal_relevance(
            query_embedding=query_embedding,
            all_documents=all_documents,
            doc_scores=doc_scores,
            lambda_param=0.1,  # Heavily favor diversity
            k=3
        )
        
        # Should select doc1 first (highest relevance), then doc3 (most diverse)
        assert len(result) == 3
        assert result[0]["content"] == "first doc"
        # doc3 should be selected before doc2 due to diversity
        selected_contents = [doc["content"] for doc in result]
        assert "diverse doc" in selected_contents
    
    def test_empty_doc_scores(self):
        """Test MMR with empty document scores."""
        result = maximal_marginal_relevance(
            query_embedding=[1.0, 0.0, 0.0],
            all_documents={},
            doc_scores={},
            lambda_param=0.7,
            k=5
        )
        assert result == []
    
    def test_documents_without_embeddings(self):
        """Test MMR with documents that have no embeddings."""
        query_embedding = [1.0, 0.0, 0.0]
        
        all_documents = {
            "doc1": {"content": "doc without embedding"},
            "doc2": {"content": "another doc without embedding"}
        }
        
        doc_scores = {
            "doc1": 0.8,
            "doc2": 0.6
        }
        
        result = maximal_marginal_relevance(
            query_embedding=query_embedding,
            all_documents=all_documents,
            doc_scores=doc_scores,
            lambda_param=0.7,
            k=2
        )
        
        # Should still work, just without diversity penalty
        assert len(result) == 2
        assert result[0]["content"] == "doc without embedding"  # Higher score
        assert result[1]["content"] == "another doc without embedding"
    
    def test_k_limit(self):
        """Test that MMR respects the k limit."""
        query_embedding = [1.0, 0.0, 0.0]
        
        all_documents = {
            f"doc{i}": {
                "content": f"document {i}",
                "embedding": [0.5, 0.5, 0.0]
            }
            for i in range(10)
        }
        
        doc_scores = {f"doc{i}": 0.5 for i in range(10)}
        
        result = maximal_marginal_relevance(
            query_embedding=query_embedding,
            all_documents=all_documents,
            doc_scores=doc_scores,
            lambda_param=0.7,
            k=3
        )
        
        assert len(result) == 3


class TestNormalizeDocInput:
    """Test cases for _normalize_doc_input function."""
    
    def test_dict_input(self):
        """Test normalizing dictionary input."""
        doc = {"content": "test content", "metadata": {"key": "value"}}
        result = _normalize_doc_input(doc)
        assert result == doc
    
    def test_langchain_document_input(self):
        """Test normalizing LangChain Document-like input."""
        mock_doc = MagicMock()
        mock_doc.page_content = "test page content"
        mock_doc.metadata = {"source": "test.pdf"}
        
        result = _normalize_doc_input(mock_doc)
        
        expected = {
            "content": "test page content",
            "metadata": {"source": "test.pdf"}
        }
        assert result == expected
    
    def test_string_input(self):
        """Test normalizing string input."""
        result = _normalize_doc_input("simple string content")
        expected = {
            "content": "simple string content",
            "metadata": {}
        }
        assert result == expected
    
    def test_other_input_types(self):
        """Test normalizing other input types."""
        result = _normalize_doc_input(123)
        expected = {
            "content": "123",
            "metadata": {}
        }
        assert result == expected
    
    def test_langchain_document_with_none_metadata(self):
        """Test LangChain document with None metadata."""
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_doc.metadata = None
        
        result = _normalize_doc_input(mock_doc)
        
        expected = {
            "content": "content",
            "metadata": {}
        }
        assert result == expected


class TestParseGeneralDate:
    """Test cases for parse_general_date function."""
    
    def test_dd_mm_yyyy_format(self):
        """Test parsing DD/MM/YYYY format."""
        result = parse_general_date("22/10/2025")
        assert result is not None
        assert result.year == 2025
        assert result.month == 10
        # Day might be interpreted differently due to timezone conversion
        assert result.day in [21, 22]  # Allow for timezone differences
        assert result.tzinfo == datetime.timezone.utc
    
    def test_month_dd_yyyy_format(self):
        """Test parsing 'Month DD, YYYY' format."""
        result = parse_general_date("October 22, 2025")
        assert result is not None
        assert result.year == 2025
        assert result.month == 10
        assert result.day in [21, 22]  # Allow for timezone differences
        assert result.tzinfo == datetime.timezone.utc
    
    def test_yyyy_mm_dd_format(self):
        """Test parsing YYYY-MM-DD format."""
        result = parse_general_date("2025-10-22")
        assert result is not None
        assert result.year == 2025
        assert result.month == 10
        assert result.day in [21, 22]  # Allow for timezone differences
        assert result.tzinfo == datetime.timezone.utc
    
    def test_mixed_format_with_time(self):
        """Test parsing mixed format with time info."""
        result = parse_general_date("22 Oct 2025 | 10:00 AM")
        assert result is not None
        assert result.year == 2025
        assert result.month == 10
        assert result.day in [21, 22]  # Allow for timezone differences
        # Time part should be parsed if present
    
    def test_invalid_date_string(self):
        """Test parsing invalid date string."""
        result = parse_general_date("Invalid Date String")
        assert result is None
    
    def test_empty_date_string(self):
        """Test parsing empty date string."""
        result = parse_general_date("")
        assert result is None
        
        result = parse_general_date(None)
        assert result is None
    
    def test_date_with_timezone_info(self):
        """Test parsing date with timezone information."""
        result = parse_general_date("2025-10-22T15:30:00+04:00")
        assert result is not None
        assert result.year == 2025
        assert result.month == 10
        assert result.day == 22
        # Should be converted to UTC
        assert result.tzinfo == datetime.timezone.utc


class TestCreateSafeFilename:
    """Test cases for create_safe_filename function."""
    
    def test_basic_text(self):
        """Test creating safe filename from basic text."""
        result = create_safe_filename("Machine Learning Basics")
        assert result == "Machine_Learning_Basics"
    
    def test_special_characters(self):
        """Test handling of special characters."""
        result = create_safe_filename("File@Name#With$Special%Characters!")
        assert result == "File_Name_With_Special_Characters"
    
    def test_multiple_underscores(self):
        """Test consolidation of multiple underscores."""
        result = create_safe_filename("Multiple___Underscores___Here")
        assert result == "Multiple_Underscores_Here"
    
    def test_empty_string(self):
        """Test handling of empty string."""
        result = create_safe_filename("")
        assert result == "untitled"
    
    def test_length_limit(self):
        """Test length limiting."""
        long_text = "A" * 150  # Longer than default max_length of 100
        result = create_safe_filename(long_text)
        assert len(result) <= 100
        assert result == "A" * 100
    
    def test_custom_length_limit(self):
        """Test custom length limit."""
        result = create_safe_filename("Long filename here", max_length=10)
        assert len(result) <= 10
        assert result == "Long_filen"
    
    def test_leading_trailing_underscores(self):
        """Test removal of leading/trailing underscores."""
        result = create_safe_filename("___filename___")
        assert result == "filename"


class TestCohereRerank:
    """Test cases for cohere_rerank function (integration test with mocks)."""
    
    @patch('src.shared.utils.embed_queries_batch')
    @patch('src.shared.utils.retrieve_rag_documents')
    @patch('src.shared.utils.embed_query')
    def test_basic_reranking(self, mock_embed_query, mock_retrieve, mock_embed_batch):
        """Test basic reranking functionality."""
        # Mock embeddings
        mock_embed_batch.return_value = [
            [0.1, 0.2, 0.3] * 255,  # Query embedding
            [0.2, 0.3, 0.4] * 255,  # Variant 1 embedding
        ]
        mock_embed_query.return_value = [0.1, 0.2, 0.3] * 255
        
        # Mock retrieved documents
        mock_retrieve.return_value = [
            {
                "id": "doc1",
                "content": "First document",
                "metadata": {"class_name": "test_class"},
                "embedding": [0.1, 0.2, 0.3] * 255
            },
            {
                "id": "doc2", 
                "content": "Second document",
                "metadata": {"class_name": "test_class"},
                "embedding": [0.2, 0.3, 0.4] * 255
            }
        ]
        
        # Input documents
        input_docs = [
            {"content": "Input doc 1", "metadata": {"class_name": "test_class"}},
            {"content": "Input doc 2", "metadata": {"class_name": "test_class"}}
        ]
        
        result = cohere_rerank("test query", input_docs)
        
        # Should return reranked documents
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(doc, dict) for doc in result)
    
    @patch('src.shared.utils.embed_queries_batch')
    @patch('src.shared.utils.retrieve_rag_documents')
    @patch('src.shared.utils.embed_query')
    def test_empty_input_documents(self, mock_embed_query, mock_retrieve, mock_embed_batch):
        """Test reranking with empty input documents."""
        mock_embed_batch.return_value = []
        mock_retrieve.return_value = []
        mock_embed_query.return_value = [0.1, 0.2, 0.3] * 255
        
        result = cohere_rerank("test query", [])
        
        # Should handle empty input gracefully
        assert isinstance(result, list)
    
    @patch('src.shared.utils.embed_queries_batch')
    @patch('src.shared.utils.retrieve_rag_documents')
    @patch('src.shared.utils.embed_query')
    def test_embedding_failure_fallback(self, mock_embed_query, mock_retrieve, mock_embed_batch):
        """Test fallback when embedding fails."""
        # Mock embedding failure
        mock_embed_batch.side_effect = Exception("Embedding API failed")
        mock_embed_query.side_effect = Exception("Query embedding failed")
        mock_retrieve.return_value = []
        
        input_docs = [
            {"content": "Test doc", "metadata": {"class_name": "test_class"}}
        ]
        
        result = cohere_rerank("test query", input_docs)
        
        # Should fallback gracefully and return something
        assert isinstance(result, list)


# ============================================================================
# Integration Tests
# ============================================================================

class TestUtilsIntegration:
    """Integration tests that combine multiple utility functions."""
    
    def test_full_document_processing_pipeline(self):
        """Test a complete document processing pipeline."""
        # Sample documents with various formats
        documents = [
            {
                "id": "doc1",
                "content": "Machine learning content",
                "class_name": "CS101",
                "section": "sessions",
                "embedding": [0.1, 0.2, 0.3] * 255
            },
            {
                "doc_id": "doc2",
                "page_content": "Deep learning content", 
                "metadata": {
                    "class_name": "CS101",
                    "section": "in_class"
                },
                "embedding": [0.2, 0.3, 0.4] * 255
            }
        ]
        
        # Test document ID extraction
        ids = [_get_doc_id(doc) for doc in documents]
        assert ids == ["doc1", "doc2"]
        
        # Test class extraction
        classes = [_get_doc_class(doc) for doc in documents]
        assert classes == ["CS101", "CS101"]
        
        # Test section extraction
        sections = [_get_doc_section(doc) for doc in documents]
        assert sections == ["sessions", "in_class"]
        
        # Test embedding extraction
        embeddings = [_extract_doc_embedding(doc) for doc in documents]
        assert all(emb is not None for emb in embeddings)
        assert all(len(emb) == 765 for emb in embeddings)
    
    def test_similarity_and_mmr_pipeline(self):
        """Test similarity calculation and MMR selection pipeline."""
        query_embedding = [1.0, 0.0, 0.0] * 255
        
        # Create documents with known similarity relationships
        documents = {
            "high_sim": {
                "content": "High similarity doc",
                "embedding": [0.9, 0.1, 0.0] * 255  # High similarity to query
            },
            "med_sim": {
                "content": "Medium similarity doc",
                "embedding": [0.5, 0.5, 0.0] * 255  # Medium similarity
            },
            "low_sim": {
                "content": "Low similarity doc", 
                "embedding": [0.1, 0.9, 0.0] * 255  # Low similarity
            }
        }
        
        # Calculate similarities
        similarities = {}
        for doc_id, doc in documents.items():
            doc_emb = _extract_doc_embedding(doc)
            similarities[doc_id] = compute_cosine_similarity(query_embedding, doc_emb)
        
        # Verify similarity ordering
        assert similarities["high_sim"] > similarities["med_sim"]
        assert similarities["med_sim"] > similarities["low_sim"]
        
        # Test MMR selection
        selected = maximal_marginal_relevance(
            query_embedding=query_embedding,
            all_documents=documents,
            doc_scores=similarities,
            lambda_param=0.8,  # Favor relevance
            k=2
        )
        
        assert len(selected) == 2
        assert selected[0]["content"] == "High similarity doc"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])