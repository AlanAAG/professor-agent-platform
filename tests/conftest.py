"""
Shared pytest fixtures for the test suite.

This module provides mock objects and fixtures to create an isolated test environment
that assumes no internet connection and no real database access.
"""

import pytest
import os
import datetime
from unittest.mock import MagicMock, patch, Mock
from typing import List, Dict, Any, Optional


# ============================================================================
# Environment Variables Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
    """Set up mock environment variables for all tests."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake_gemini_key_for_testing")
    monkeypatch.setenv("SUPABASE_URL", "https://fake-supabase-url.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "fake_supabase_key_for_testing")
    monkeypatch.setenv("EXTERNAL_SUPABASE_URL", "https://fake-external-supabase.supabase.co")
    monkeypatch.setenv("EXTERNAL_SUPABASE_SERVICE_KEY", "fake_external_key")
    monkeypatch.setenv("COACH_USERNAME", "test_coach")
    monkeypatch.setenv("SECRET_API_KEY", "fake_secret_api_key")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "models/embedding-001")


# ============================================================================
# Gemini API Mocks
# ============================================================================

@pytest.fixture
def mock_genai():
    """Mock the google.generativeai module."""
    with patch('google.generativeai.configure') as mock_configure, \
         patch('google.generativeai.embed_content') as mock_embed_content:
        
        # Mock embed_content to return deterministic embeddings
        mock_embed_content.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 153  # 765 dimensions (typical for Gemini)
        }
        
        yield {
            'configure': mock_configure,
            'embed_content': mock_embed_content
        }


@pytest.fixture
def mock_gemini_model():
    """Mock the Gemini chat model for LangChain."""
    mock_model = MagicMock()
    
    # Mock the invoke method to return deterministic responses
    mock_model.invoke.return_value = MagicMock(content="This is a cleaned transcript with proper punctuation and formatting.")
    
    # Mock the generate_content method for direct API calls
    mock_response = MagicMock()
    mock_response.text = "This is a generated response from the mock model."
    mock_model.generate_content.return_value = mock_response
    
    return mock_model


# ============================================================================
# Supabase Mocks
# ============================================================================

@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client with common database operations."""
    mock_client = MagicMock()
    
    # Mock RPC calls (for match_documents)
    mock_rpc_response = MagicMock()
    mock_rpc_response.data = [
        {
            "id": "doc_1",
            "content": "Sample document content 1",
            "metadata": {
                "class_name": "test_class",
                "content_type": "lecture",
                "title": "Test Document 1",
                "similarity": 0.85
            },
            "embedding": [0.1, 0.2, 0.3] * 255  # Mock embedding
        },
        {
            "id": "doc_2", 
            "content": "Sample document content 2",
            "metadata": {
                "class_name": "test_class",
                "content_type": "reading",
                "title": "Test Document 2",
                "similarity": 0.78
            },
            "embedding": [0.2, 0.3, 0.4] * 255  # Mock embedding
        }
    ]
    mock_client.rpc.return_value.execute.return_value = mock_rpc_response
    
    # Mock table operations
    mock_table = MagicMock()
    mock_select_response = MagicMock()
    mock_select_response.data = []
    mock_select_response.count = 0
    
    # Chain the table operations
    mock_table.select.return_value = mock_table
    mock_table.eq.return_value = mock_table
    mock_table.contains.return_value = mock_table
    mock_table.gt.return_value = mock_table
    mock_table.limit.return_value = mock_table
    mock_table.execute.return_value = mock_select_response
    
    mock_client.table.return_value = mock_table
    
    return mock_client


@pytest.fixture
def mock_supabase_with_existing_docs(mock_supabase_client):
    """Mock Supabase client that returns existing documents."""
    # Modify the response to indicate documents exist
    mock_response = MagicMock()
    mock_response.count = 2
    mock_response.data = [{"id": "existing_doc_1"}, {"id": "existing_doc_2"}]
    
    mock_supabase_client.table.return_value.execute.return_value = mock_response
    return mock_supabase_client


# ============================================================================
# Selenium WebDriver Mocks
# ============================================================================

@pytest.fixture
def mock_webdriver():
    """Mock Selenium WebDriver with common browser operations."""
    mock_driver = MagicMock()
    
    # Mock driver methods
    mock_driver.get = MagicMock()
    mock_driver.quit = MagicMock()
    mock_driver.close = MagicMock()
    mock_driver.execute_script = MagicMock(return_value="Mock script result")
    
    # Mock element finding
    mock_element = MagicMock()
    mock_element.text = "Sample element text"
    mock_element.get_attribute.return_value = "sample_attribute_value"
    mock_element.click = MagicMock()
    
    mock_driver.find_element.return_value = mock_element
    mock_driver.find_elements.return_value = [mock_element, mock_element]
    
    # Mock page source and current URL
    mock_driver.page_source = "<html><body>Mock page content</body></html>"
    mock_driver.current_url = "https://mock-url.com"
    
    return mock_driver


# ============================================================================
# PDF Processing Mocks (PyMuPDF/fitz)
# ============================================================================

@pytest.fixture
def mock_fitz():
    """Mock PyMuPDF (fitz) for PDF processing."""
    with patch('fitz.open') as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        
        # Mock page content
        mock_page.get_text.return_value = "Sample PDF page text content"
        mock_page.get_images.return_value = [
            {"xref": 1, "width": 100, "height": 100}
        ]
        mock_page.get_links.return_value = [
            {"uri": "https://example.com", "page": 0}
        ]
        
        # Mock document structure
        mock_doc.load_page.return_value = mock_page
        mock_doc.page_count = 3
        mock_doc.__len__ = lambda self: 3
        mock_doc.__iter__ = lambda self: iter([mock_page, mock_page, mock_page])
        
        mock_open.return_value = mock_doc
        
        yield {
            'open': mock_open,
            'doc': mock_doc,
            'page': mock_page
        }


# ============================================================================
# HTTP Requests Mocks
# ============================================================================

@pytest.fixture
def mock_requests():
    """Mock requests library for HTTP operations."""
    with patch('requests.get') as mock_get, \
         patch('requests.head') as mock_head, \
         patch('requests.post') as mock_post:
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Mock response content"
        mock_response.content = b"Mock binary content"
        mock_response.headers = {
            'content-type': 'application/pdf',
            'content-length': '12345'
        }
        mock_response.json.return_value = {"status": "success", "data": "mock_data"}
        
        mock_get.return_value = mock_response
        mock_head.return_value = mock_response
        mock_post.return_value = mock_response
        
        yield {
            'get': mock_get,
            'head': mock_head,
            'post': mock_post,
            'response': mock_response
        }


# ============================================================================
# LangChain Mocks
# ============================================================================

@pytest.fixture
def mock_text_splitter():
    """Mock LangChain text splitter."""
    mock_splitter = MagicMock()
    
    # Mock Document class
    mock_document = MagicMock()
    mock_document.page_content = "Sample chunk content"
    mock_document.metadata = {}
    
    # Mock create_documents to return list of mock documents
    mock_splitter.create_documents.return_value = [
        mock_document,
        mock_document,
        mock_document
    ]
    
    return mock_splitter


@pytest.fixture
def mock_vector_store():
    """Mock LangChain vector store."""
    mock_store = MagicMock()
    
    # Mock add_documents method
    mock_store.add_documents.return_value = ["doc_id_1", "doc_id_2", "doc_id_3"]
    
    # Mock similarity search
    mock_store.similarity_search.return_value = [
        MagicMock(page_content="Similar doc 1", metadata={"score": 0.9}),
        MagicMock(page_content="Similar doc 2", metadata={"score": 0.8})
    ]
    
    return mock_store


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "class_name": "test_class",
        "content_type": "lecture",
        "source_file": "test_document.pdf",
        "source_url": "https://example.com/test_document.pdf",
        "title": "Test Document Title",
        "lecture_date": "2025-10-28",
        "page_number": 1,
        "retrieval_date": datetime.datetime.now().isoformat()
    }


@pytest.fixture
def sample_documents():
    """Sample document data for testing."""
    return [
        {
            "id": "doc_1",
            "content": "This is the first test document content.",
            "metadata": {
                "class_name": "test_class",
                "content_type": "lecture",
                "title": "Document 1"
            },
            "embedding": [0.1, 0.2, 0.3] * 255
        },
        {
            "id": "doc_2", 
            "content": "This is the second test document content.",
            "metadata": {
                "class_name": "test_class",
                "content_type": "reading", 
                "title": "Document 2"
            },
            "embedding": [0.2, 0.3, 0.4] * 255
        }
    ]


@pytest.fixture
def sample_raw_transcript():
    """Sample raw transcript text for cleaning tests."""
    return """
    0:01 welcome to today's lecture
    0:15 we will be discussing machine learning
    1:30 machine learning is a subset of artificial intelligence
    2:45 it involves algorithms that can learn from data
    3:00 without being explicitly programmed
    """


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing similarity calculations."""
    return {
        "query_embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 153,
        "doc_embeddings": [
            [0.1, 0.2, 0.3, 0.4, 0.5] * 153,  # High similarity
            [0.9, 0.8, 0.7, 0.6, 0.5] * 153,  # Low similarity
            [0.2, 0.3, 0.4, 0.5, 0.6] * 153   # Medium similarity
        ]
    }


# ============================================================================
# Utility Functions for Tests
# ============================================================================

def create_mock_document(doc_id: str = "test_doc", content: str = "test content", **metadata_kwargs):
    """Helper function to create mock document dictionaries."""
    base_metadata = {
        "class_name": "test_class",
        "content_type": "test_type"
    }
    base_metadata.update(metadata_kwargs)
    
    return {
        "id": doc_id,
        "content": content,
        "page_content": content,  # For LangChain compatibility
        "metadata": base_metadata
    }


def create_mock_langchain_document(content: str = "test content", **metadata_kwargs):
    """Helper function to create mock LangChain Document objects."""
    mock_doc = MagicMock()
    mock_doc.page_content = content
    mock_doc.metadata = metadata_kwargs
    return mock_doc