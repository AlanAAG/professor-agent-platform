"""
Unit tests for src/refinery/embedding.py

Tests cover embedding functions with various edge cases and scenarios,
using mocked dependencies to ensure isolated testing environment.
"""

import pytest
import datetime
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List

# Import the functions we want to test
from src.refinery.embedding import (
    validate_metadata,
    chunk_and_embed_text,
    check_if_embedded,
    check_if_embedded_recently,
    url_exists_in_db,
    check_if_embedded_recently_sync,
    url_exists_in_db_sync,
    REQUIRED_METADATA,
    OPTIONAL_METADATA
)


class TestValidateMetadata:
    """Test cases for validate_metadata function."""
    
    def test_valid_metadata_with_required_fields(self):
        """Test validation of metadata with all required fields."""
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture"
        }
        
        result = validate_metadata(metadata)
        
        assert result["class_name"] == "CS101"
        assert result["content_type"] == "lecture"
        assert "retrieval_date" in result  # Should be added automatically
    
    def test_valid_metadata_with_optional_fields(self):
        """Test validation of metadata with optional fields."""
        metadata = {
            "class_name": "MATH201",
            "content_type": "reading",
            "source_file": "textbook.pdf",
            "source_url": "https://example.com/textbook.pdf",
            "title": "Linear Algebra Basics",
            "lecture_date": "2025-10-28",
            "page_number": 42,
            "links": ["https://example.com/ref1", "https://example.com/ref2"]
        }
        
        result = validate_metadata(metadata)
        
        # All fields should be preserved
        for key, value in metadata.items():
            assert result[key] == value
        
        # retrieval_date should be added
        assert "retrieval_date" in result
    
    def test_missing_required_field_class_name(self):
        """Test validation fails when class_name is missing."""
        metadata = {
            "content_type": "lecture"
            # Missing class_name
        }
        
        with pytest.raises(ValueError, match="Missing required metadata field: class_name"):
            validate_metadata(metadata)
    
    def test_missing_required_field_content_type(self):
        """Test validation fails when content_type is missing."""
        metadata = {
            "class_name": "CS101"
            # Missing content_type
        }
        
        with pytest.raises(ValueError, match="Missing required metadata field: content_type"):
            validate_metadata(metadata)
    
    def test_none_values_removed(self):
        """Test that None values are removed from metadata."""
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture",
            "source_file": None,
            "title": "Valid Title",
            "page_number": None,
            "links": ["valid_link"]
        }
        
        result = validate_metadata(metadata)
        
        # None values should be removed
        assert "source_file" not in result
        assert "page_number" not in result
        
        # Valid values should be preserved
        assert result["class_name"] == "CS101"
        assert result["content_type"] == "lecture"
        assert result["title"] == "Valid Title"
        assert result["links"] == ["valid_link"]
    
    def test_retrieval_date_auto_added(self):
        """Test that retrieval_date is automatically added."""
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture"
        }
        
        before_time = datetime.datetime.now()
        result = validate_metadata(metadata)
        after_time = datetime.datetime.now()
        
        assert "retrieval_date" in result
        
        # Parse the retrieval_date and verify it's reasonable
        retrieval_dt = datetime.datetime.fromisoformat(result["retrieval_date"])
        assert before_time <= retrieval_dt <= after_time
    
    def test_existing_retrieval_date_preserved(self):
        """Test that existing retrieval_date is preserved."""
        custom_date = "2025-01-01T12:00:00"
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture",
            "retrieval_date": custom_date
        }
        
        result = validate_metadata(metadata)
        
        assert result["retrieval_date"] == custom_date
    
    def test_invalid_metadata_type(self):
        """Test validation fails with non-dictionary input."""
        with pytest.raises(ValueError, match="Metadata must be a dictionary"):
            validate_metadata("not_a_dict")
        
        with pytest.raises(ValueError, match="Metadata must be a dictionary"):
            validate_metadata(None)
        
        with pytest.raises(ValueError, match="Metadata must be a dictionary"):
            validate_metadata(123)


class TestChunkAndEmbedText:
    """Test cases for chunk_and_embed_text function."""
    
    @patch('src.refinery.embedding.vector_store')
    @patch('src.refinery.embedding.text_splitter')
    def test_successful_chunking_and_embedding(self, mock_text_splitter, mock_vector_store):
        """Test successful text chunking and embedding."""
        # Mock text splitter
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {}
        mock_text_splitter.create_documents.return_value = [mock_doc1, mock_doc2]
        
        # Mock vector store
        mock_vector_store.add_documents.return_value = ["chunk_id_1", "chunk_id_2"]
        
        clean_text = "This is a long piece of text that will be chunked into smaller pieces for embedding."
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture",
            "title": "Test Lecture"
        }
        
        # Should not raise any exceptions
        chunk_and_embed_text(clean_text, metadata)
        
        # Verify text splitter was called
        mock_text_splitter.create_documents.assert_called_once_with([clean_text])
        
        # Verify metadata was attached to documents
        assert mock_doc1.metadata["class_name"] == "CS101"
        assert mock_doc1.metadata["content_type"] == "lecture"
        assert mock_doc1.metadata["title"] == "Test Lecture"
        assert "retrieval_date" in mock_doc1.metadata
        
        # Verify vector store was called
        mock_vector_store.add_documents.assert_called_once_with([mock_doc1, mock_doc2])
    
    @patch('src.refinery.embedding.vector_store', None)
    def test_vector_store_not_initialized(self):
        """Test behavior when vector store is not initialized."""
        clean_text = "Test text"
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture"
        }
        
        with pytest.raises(EnvironmentError, match="Vector store is not initialized"):
            chunk_and_embed_text(clean_text, metadata)
    
    def test_empty_text_handling(self):
        """Test handling of empty text input."""
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture"
        }
        
        # Should handle gracefully and log info
        with patch('src.refinery.embedding.vector_store') as mock_vector_store:
            chunk_and_embed_text("", metadata)
            chunk_and_embed_text(None, metadata)
            
            # Vector store should not be called
            mock_vector_store.add_documents.assert_not_called()
    
    @patch('src.refinery.embedding.vector_store')
    @patch('src.refinery.embedding.text_splitter')
    def test_metadata_validation_integration(self, mock_text_splitter, mock_vector_store):
        """Test that metadata validation is properly integrated."""
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_text_splitter.create_documents.return_value = [mock_doc]
        mock_vector_store.add_documents.return_value = ["chunk_id"]
        
        clean_text = "Test text for metadata validation"
        
        # Missing required field should raise error
        invalid_metadata = {"class_name": "CS101"}  # Missing content_type
        
        with pytest.raises(ValueError, match="Missing required metadata field: content_type"):
            chunk_and_embed_text(clean_text, invalid_metadata)
    
    @patch('src.refinery.embedding.vector_store')
    @patch('src.refinery.embedding.text_splitter')
    def test_retry_mechanism_on_failure(self, mock_text_splitter, mock_vector_store):
        """Test retry mechanism when vector store operations fail."""
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_text_splitter.create_documents.return_value = [mock_doc]
        
        # Mock vector store to fail initially, then succeed
        mock_vector_store.add_documents.side_effect = [
            ConnectionError("Network error"),
            ConnectionError("Network error"),
            ["chunk_id"]  # Success on third try
        ]
        
        clean_text = "Test text for retry mechanism"
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture"
        }
        
        # Should eventually succeed after retries
        chunk_and_embed_text(clean_text, metadata)
        
        # Should have been called 3 times (2 failures + 1 success)
        assert mock_vector_store.add_documents.call_count == 3
    
    @patch('src.refinery.embedding.vector_store')
    @patch('src.refinery.embedding.text_splitter')
    def test_retry_exhaustion_raises_exception(self, mock_text_splitter, mock_vector_store):
        """Test that exception is raised when all retries are exhausted."""
        mock_doc = MagicMock()
        mock_doc.metadata = {}
        mock_text_splitter.create_documents.return_value = [mock_doc]
        
        # Mock vector store to always fail
        mock_vector_store.add_documents.side_effect = ConnectionError("Persistent network error")
        
        clean_text = "Test text for retry exhaustion"
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture"
        }
        
        # Should raise RetryError (from tenacity) after all retries are exhausted
        from tenacity import RetryError
        with pytest.raises(RetryError):
            chunk_and_embed_text(clean_text, metadata)
    
    @patch('src.refinery.embedding.vector_store')
    @patch('src.refinery.embedding.text_splitter')
    def test_logging_behavior(self, mock_text_splitter, mock_vector_store, caplog):
        """Test that appropriate log messages are generated."""
        import logging
        
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {}
        mock_text_splitter.create_documents.return_value = [mock_doc1, mock_doc2]
        mock_vector_store.add_documents.return_value = ["chunk_id_1", "chunk_id_2"]
        
        clean_text = "Test text for logging verification"
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture"
        }
        
        with caplog.at_level(logging.INFO):
            chunk_and_embed_text(clean_text, metadata)
        
        # Should log chunking and embedding success
        log_messages = [record.message for record in caplog.records]
        assert any("Chunking and embedding" in msg for msg in log_messages)
        assert any("Split text into" in msg for msg in log_messages)
        assert any("Embedding successful" in msg for msg in log_messages)


class TestCheckIfEmbedded:
    """Test cases for check_if_embedded function."""
    
    def test_always_returns_false(self):
        """Test that check_if_embedded always returns False (as documented)."""
        filter_dict = {
            "class_name": "CS101",
            "content_type": "lecture"
        }
        
        result = check_if_embedded(filter_dict)
        assert result is False
        
        # Should work with any filter
        result = check_if_embedded({})
        assert result is False
        
        result = check_if_embedded({"any": "filter"})
        assert result is False


class TestCheckIfEmbeddedRecently:
    """Test cases for check_if_embedded_recently function."""
    
    @pytest.mark.asyncio
    @patch('src.refinery.embedding.supabase')
    async def test_recent_document_found(self, mock_supabase):
        """Test finding recently embedded documents."""
        # Mock Supabase response indicating documents exist
        mock_response = MagicMock()
        mock_response.count = 2
        
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_response
        mock_query.eq.return_value = mock_query
        mock_query.gt.return_value = mock_query
        mock_query.limit.return_value = mock_query
        
        mock_table = MagicMock()
        mock_table.select.return_value = mock_query
        
        mock_supabase.table.return_value = mock_table
        
        filter_dict = {
            "source_url": "https://example.com/document.pdf"
        }
        
        result = await check_if_embedded_recently(filter_dict, days=7)
        
        assert result is True
        mock_supabase.table.assert_called_with("documents")
        mock_query.eq.assert_called_with("metadata->>source_url", "https://example.com/document.pdf")
    
    @pytest.mark.asyncio
    @patch('src.refinery.embedding.supabase')
    async def test_no_recent_documents(self, mock_supabase):
        """Test when no recent documents are found."""
        # Mock Supabase response indicating no documents
        mock_response = MagicMock()
        mock_response.count = 0
        
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_response
        mock_query.contains.return_value = mock_query
        mock_query.gt.return_value = mock_query
        mock_query.limit.return_value = mock_query
        
        mock_table = MagicMock()
        mock_table.select.return_value = mock_query
        
        mock_supabase.table.return_value = mock_table
        
        filter_dict = {
            "class_name": "CS101",
            "content_type": "lecture"
        }
        
        result = await check_if_embedded_recently(filter_dict, days=7)
        
        assert result is False
        mock_query.contains.assert_called_with("metadata", filter_dict)
    
    @pytest.mark.asyncio
    @patch('src.refinery.embedding.supabase', None)
    async def test_supabase_not_configured(self):
        """Test behavior when Supabase is not configured."""
        filter_dict = {"class_name": "CS101"}
        
        result = await check_if_embedded_recently(filter_dict)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_invalid_filter_type(self):
        """Test behavior with invalid filter type."""
        result = await check_if_embedded_recently("invalid_filter")
        assert result is False
        
        result = await check_if_embedded_recently(None)
        assert result is False
    
    @pytest.mark.asyncio
    @patch('src.refinery.embedding.supabase')
    async def test_database_error_handling(self, mock_supabase):
        """Test handling of database errors."""
        # Mock Supabase to raise an exception
        mock_supabase.table.side_effect = Exception("Database connection error")
        
        filter_dict = {"class_name": "CS101"}
        
        result = await check_if_embedded_recently(filter_dict)
        
        # Should return False on error (fail-open)
        assert result is False
    
    @pytest.mark.asyncio
    @patch('src.refinery.embedding.supabase')
    async def test_custom_days_parameter(self, mock_supabase):
        """Test custom days parameter for recency check."""
        mock_response = MagicMock()
        mock_response.count = 1
        
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_response
        mock_query.contains.return_value = mock_query
        mock_query.gt.return_value = mock_query
        mock_query.limit.return_value = mock_query
        
        mock_table = MagicMock()
        mock_table.select.return_value = mock_query
        
        mock_supabase.table.return_value = mock_table
        
        filter_dict = {"class_name": "CS101"}
        
        # Test with custom days parameter
        result = await check_if_embedded_recently(filter_dict, days=14)
        
        assert result is True
        
        # Verify the date calculation was done with 14 days
        call_args = mock_query.gt.call_args
        assert call_args[0][0] == "metadata->>retrieval_date"
        # The actual date string would be calculated based on 14 days ago


class TestUrlExistsInDb:
    """Test cases for url_exists_in_db function."""
    
    @pytest.mark.asyncio
    @patch('src.refinery.embedding.supabase')
    async def test_url_exists(self, mock_supabase):
        """Test finding existing URL in database."""
        # Mock Supabase response indicating URL exists
        mock_response = MagicMock()
        mock_response.count = 1
        
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_response
        mock_query.eq.return_value = mock_query
        mock_query.limit.return_value = mock_query
        
        mock_table = MagicMock()
        mock_table.select.return_value = mock_query
        
        mock_supabase.table.return_value = mock_table
        
        url = "https://example.com/document.pdf"
        result = await url_exists_in_db(url)
        
        assert result is True
        mock_query.eq.assert_called_with("metadata->>source_url", url)
    
    @pytest.mark.asyncio
    @patch('src.refinery.embedding.supabase')
    async def test_url_does_not_exist(self, mock_supabase):
        """Test when URL does not exist in database."""
        # Mock Supabase response indicating URL doesn't exist
        mock_response = MagicMock()
        mock_response.count = 0
        
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_response
        mock_query.eq.return_value = mock_query
        mock_query.limit.return_value = mock_query
        
        mock_table = MagicMock()
        mock_table.select.return_value = mock_query
        
        mock_supabase.table.return_value = mock_table
        
        url = "https://example.com/nonexistent.pdf"
        result = await url_exists_in_db(url)
        
        assert result is False
    
    @pytest.mark.asyncio
    @patch('src.refinery.embedding.supabase', None)
    async def test_supabase_not_configured(self):
        """Test behavior when Supabase is not configured."""
        url = "https://example.com/document.pdf"
        result = await url_exists_in_db(url)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_empty_url(self):
        """Test behavior with empty URL."""
        result = await url_exists_in_db("")
        assert result is False
        
        result = await url_exists_in_db(None)
        assert result is False
    
    @pytest.mark.asyncio
    @patch('src.refinery.embedding.supabase')
    async def test_database_error_handling(self, mock_supabase):
        """Test handling of database errors."""
        # Mock Supabase to raise an exception
        mock_supabase.table.side_effect = Exception("Database error")
        
        url = "https://example.com/document.pdf"
        result = await url_exists_in_db(url)
        
        # Should return False on error (fail-open)
        assert result is False


class TestSynchronousHelpers:
    """Test cases for synchronous helper functions."""
    
    @patch('src.refinery.embedding.supabase')
    def test_check_if_embedded_recently_sync(self, mock_supabase):
        """Test synchronous version of check_if_embedded_recently."""
        # Mock Supabase response
        mock_response = MagicMock()
        mock_response.count = 1
        
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_response
        mock_query.contains.return_value = mock_query
        mock_query.gt.return_value = mock_query
        mock_query.limit.return_value = mock_query
        
        mock_table = MagicMock()
        mock_table.select.return_value = mock_query
        
        mock_supabase.table.return_value = mock_table
        
        filter_dict = {"class_name": "CS101"}
        result = check_if_embedded_recently_sync(filter_dict, days=2)
        
        assert result is True
        mock_supabase.table.assert_called_with("documents")
    
    @patch('src.refinery.embedding.supabase')
    def test_url_exists_in_db_sync(self, mock_supabase):
        """Test synchronous version of url_exists_in_db."""
        # Mock Supabase response
        mock_response = MagicMock()
        mock_response.count = 1
        
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_response
        mock_query.eq.return_value = mock_query
        mock_query.limit.return_value = mock_query
        
        mock_table = MagicMock()
        mock_table.select.return_value = mock_query
        
        mock_supabase.table.return_value = mock_table
        
        url = "https://example.com/document.pdf"
        result = url_exists_in_db_sync(url)
        
        assert result is True
        mock_query.eq.assert_called_with("metadata->>source_url", url)
    
    @patch('src.refinery.embedding.supabase', None)
    def test_sync_functions_without_supabase(self):
        """Test synchronous functions when Supabase is not configured."""
        result = check_if_embedded_recently_sync({"class_name": "CS101"})
        assert result is False
        
        result = url_exists_in_db_sync("https://example.com/doc.pdf")
        assert result is False


class TestEmbeddingIntegration:
    """Integration tests combining multiple embedding functions."""
    
    @patch('src.refinery.embedding.vector_store')
    @patch('src.refinery.embedding.text_splitter')
    @patch('src.refinery.embedding.supabase')
    def test_complete_embedding_workflow(self, mock_supabase, mock_text_splitter, mock_vector_store):
        """Test complete workflow from text to embedded chunks."""
        # Setup mocks
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {}
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {}
        mock_text_splitter.create_documents.return_value = [mock_doc1, mock_doc2]
        mock_vector_store.add_documents.return_value = ["chunk_1", "chunk_2"]
        
        # Mock Supabase to indicate no recent documents
        mock_response = MagicMock()
        mock_response.count = 0
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_response
        mock_query.eq.return_value = mock_query
        mock_query.gt.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_table = MagicMock()
        mock_table.select.return_value = mock_query
        mock_supabase.table.return_value = mock_table
        
        # Test data
        clean_text = "This is a comprehensive test of the embedding workflow with multiple sentences and concepts."
        metadata = {
            "class_name": "CS101",
            "content_type": "lecture",
            "source_url": "https://example.com/lecture.pdf",
            "title": "Machine Learning Basics"
        }
        
        # Check if already embedded (should return False)
        url_exists = url_exists_in_db_sync(metadata["source_url"])
        assert url_exists is False
        
        # Perform embedding
        chunk_and_embed_text(clean_text, metadata)
        
        # Verify the workflow
        mock_text_splitter.create_documents.assert_called_once_with([clean_text])
        mock_vector_store.add_documents.assert_called_once()
        
        # Verify metadata was properly validated and attached
        embedded_metadata = mock_doc1.metadata
        assert embedded_metadata["class_name"] == "CS101"
        assert embedded_metadata["content_type"] == "lecture"
        assert embedded_metadata["source_url"] == "https://example.com/lecture.pdf"
        assert embedded_metadata["title"] == "Machine Learning Basics"
        assert "retrieval_date" in embedded_metadata
    
    def test_metadata_validation_edge_cases(self):
        """Test edge cases in metadata validation."""
        # Test with all optional fields as None
        metadata_with_nones = {
            "class_name": "CS101",
            "content_type": "lecture",
            "source_file": None,
            "source_url": None,
            "title": None,
            "lecture_date": None,
            "page_number": None,
            "links": None
        }
        
        result = validate_metadata(metadata_with_nones)
        
        # Only required fields and retrieval_date should remain
        expected_keys = {"class_name", "content_type", "retrieval_date"}
        assert set(result.keys()) == expected_keys
        
        # Test with mixed valid and None values
        mixed_metadata = {
            "class_name": "MATH201",
            "content_type": "assignment",
            "source_file": "homework.pdf",
            "source_url": None,
            "title": "Homework 5",
            "page_number": None,
            "links": ["https://example.com/ref"]
        }
        
        result = validate_metadata(mixed_metadata)
        
        # Should preserve non-None values
        assert result["class_name"] == "MATH201"
        assert result["content_type"] == "assignment"
        assert result["source_file"] == "homework.pdf"
        assert result["title"] == "Homework 5"
        assert result["links"] == ["https://example.com/ref"]
        
        # Should remove None values
        assert "source_url" not in result
        assert "page_number" not in result
    
    @patch('src.refinery.embedding.supabase')
    def test_database_query_patterns(self, mock_supabase):
        """Test different database query patterns used by the module."""
        # Setup common mock structure
        mock_response = MagicMock()
        mock_query = MagicMock()
        mock_query.execute.return_value = mock_response
        mock_query.eq.return_value = mock_query
        mock_query.contains.return_value = mock_query
        mock_query.gt.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_table = MagicMock()
        mock_table.select.return_value = mock_query
        mock_supabase.table.return_value = mock_table
        
        # Test URL-based query
        mock_response.count = 1
        result = url_exists_in_db_sync("https://example.com/doc.pdf")
        assert result is True
        mock_query.eq.assert_called_with("metadata->>source_url", "https://example.com/doc.pdf")
        
        # Test metadata-based query
        mock_response.count = 0
        filter_dict = {"class_name": "CS101", "content_type": "lecture"}
        result = check_if_embedded_recently_sync(filter_dict)
        assert result is False
        mock_query.contains.assert_called_with("metadata", filter_dict)
        
        # Verify date filtering was applied
        mock_query.gt.assert_called()
        gt_call_args = mock_query.gt.call_args
        assert gt_call_args[0][0] == "metadata->>retrieval_date"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])