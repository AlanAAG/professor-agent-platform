"""
Unit tests for src/refinery/cleaning.py

Tests cover transcript cleaning functions with various edge cases and scenarios,
using mocked LLM dependencies to ensure isolated testing environment.
"""

import pytest
from unittest.mock import patch, MagicMock
import logging

# Import the functions we want to test
from src.refinery.cleaning import (
    _clean_transcript_locally,
    _get_cleaning_prompt,
    clean_transcript_with_llm
)


class TestCleanTranscriptLocally:
    """Test cases for _clean_transcript_locally function."""
    
    def test_basic_transcript_cleaning(self):
        """Test basic transcript cleaning functionality."""
        raw_text = """
        0:01 welcome to today's lecture
        0:15 we will be discussing machine learning
        1:30 machine learning is a subset of artificial intelligence
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # Should remove timestamps
        assert "0:01" not in result
        assert "0:15" not in result
        assert "1:30" not in result
        
        # Should preserve content
        assert "welcome to today's lecture" in result.lower()
        assert "machine learning" in result.lower()
        assert "artificial intelligence" in result.lower()
        
        # Should add punctuation
        assert result.endswith(".")
    
    def test_standalone_timestamp_removal(self):
        """Test removal of standalone timestamps on their own lines."""
        raw_text = """
        0:01
        Welcome to the lecture
        1:30
        Today we discuss AI
        2:45
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # Timestamps on their own lines should be removed
        assert "0:01" not in result
        assert "1:30" not in result
        assert "2:45" not in result
        
        # Content should be preserved
        assert "Welcome to the lecture" in result
        assert "Today we discuss AI" in result
    
    def test_inline_timestamp_removal(self):
        """Test removal of inline timestamps surrounded by whitespace."""
        raw_text = """
        Welcome 0:01 to today's lecture about 1:30 machine learning and 2:45 artificial intelligence
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # Inline timestamps should be removed
        assert "0:01" not in result
        assert "1:30" not in result
        assert "2:45" not in result
        
        # Text should flow naturally
        assert "Welcome to today's lecture" in result
        assert "machine learning and artificial intelligence" in result
    
    def test_mixed_timestamp_formats(self):
        """Test handling of different timestamp formats."""
        raw_text = """
        0:01 Introduction
        10:30 Main topic
        1:05:30 Advanced concepts
        59:59 Conclusion
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # All timestamp formats should be removed
        assert "0:01" not in result
        assert "10:30" not in result
        assert "1:05:30" not in result
        assert "59:59" not in result
        
        # Content should be preserved
        assert "Introduction" in result
        assert "Main topic" in result
        assert "Advanced concepts" in result
        assert "Conclusion" in result
    
    def test_whitespace_normalization(self):
        """Test normalization of excessive whitespace."""
        raw_text = """
        
        
        This    has     excessive     whitespace
        
        
        
        And multiple blank lines
        
        
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # Should normalize whitespace
        assert "This has excessive whitespace" in result
        assert "And multiple blank lines" in result
        
        # Should not have excessive blank lines
        assert "\n\n\n" not in result
    
    def test_punctuation_addition(self):
        """Test addition of terminal punctuation."""
        raw_text = """
        This sentence has no punctuation
        This one ends with a question mark?
        This one already has a period.
        This one has an exclamation!
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # Should add punctuation where missing
        lines = result.split('\n')
        for line in lines:
            if line.strip():
                assert line.strip().endswith(('.', '!', '?'))
    
    def test_capitalization_fix(self):
        """Test basic capitalization fixes."""
        raw_text = """
        this sentence should start with capital. and this one too.
        what about questions? they should be capitalized too!
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # Should capitalize sentence starts
        assert result.startswith("This sentence")
        assert ". And this one" in result
        assert "? They should" in result
    
    def test_paragraph_formation(self):
        """Test formation of logical paragraphs."""
        raw_text = """
        First concept here
        More about first concept
        ok
        Second concept starts
        Details about second concept
        um
        Third concept
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # Should form paragraphs (multiple lines joined)
        paragraphs = result.split('\n\n')
        assert len(paragraphs) >= 2  # Should create multiple paragraphs
        
        # Short fragments like "ok" and "um" should cause breaks
        for paragraph in paragraphs:
            if paragraph.strip():
                # Each paragraph should be substantial
                assert len(paragraph.strip().split()) > 2 or paragraph.strip() in ["ok", "um"]
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = _clean_transcript_locally("")
        assert result == ""
        
        result = _clean_transcript_locally(None)
        assert result == ""
    
    def test_whitespace_only_input(self):
        """Test handling of whitespace-only input."""
        result = _clean_transcript_locally("   \n\n   \t   ")
        assert result == ""
    
    def test_timestamps_with_seconds(self):
        """Test handling of timestamps with seconds."""
        raw_text = """
        0:01:30 Welcome to the lecture
        0:02:45 First topic
        1:15:20 Second topic
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # All timestamp formats should be removed
        assert "0:01:30" not in result
        assert "0:02:45" not in result
        assert "1:15:20" not in result
        
        # Content should be preserved
        assert "Welcome to the lecture" in result
        assert "First topic" in result
        assert "Second topic" in result
    
    def test_preserve_numbers_that_arent_timestamps(self):
        """Test that numbers that aren't timestamps are preserved."""
        raw_text = """
        The year 2025 is important
        We have 3:2 ratio here
        Chapter 1:1 discusses this
        At 3:30 we start  
        """
        
        result = _clean_transcript_locally(raw_text)
        
        # Should preserve non-timestamp numbers
        assert "2025" in result
        assert "3:2 ratio" in result
        assert "1:1" in result  # This might be edge case
        
        # Should remove actual timestamp
        assert "3:30" not in result


class TestGetCleaningPrompt:
    """Test cases for _get_cleaning_prompt function."""
    
    def test_prompt_template_creation(self):
        """Test that cleaning prompt template is created correctly."""
        prompt = _get_cleaning_prompt()
        
        # Should be a ChatPromptTemplate
        assert hasattr(prompt, 'format_messages') or hasattr(prompt, 'format')
        
        # Should contain expected instructions
        template_str = str(prompt)
        assert "transcript editor" in template_str.lower()
        assert "remove all timestamps" in template_str.lower()
        assert "fix punctuation" in template_str.lower()
        assert "create paragraphs" in template_str.lower()
    
    def test_prompt_formatting(self):
        """Test that prompt can be formatted with raw text."""
        prompt = _get_cleaning_prompt()
        
        # Should be able to format with raw_text parameter
        try:
            if hasattr(prompt, 'format'):
                formatted = prompt.format(raw_text="test transcript")
                assert "test transcript" in formatted
            elif hasattr(prompt, 'format_messages'):
                messages = prompt.format_messages(raw_text="test transcript")
                assert any("test transcript" in str(msg) for msg in messages)
        except Exception as e:
            pytest.fail(f"Prompt formatting failed: {e}")


class TestCleanTranscriptWithLlm:
    """Test cases for clean_transcript_with_llm function."""
    
    @patch('src.refinery.cleaning.model')
    def test_successful_llm_cleaning(self, mock_model):
        """Test successful LLM-based transcript cleaning."""
        # Mock the LLM chain to return a string directly
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "This is a cleaned transcript with proper punctuation and formatting."
        
        # Create a proper chain mock that supports the | operator
        mock_prompt = MagicMock()
        mock_parser = MagicMock()
        
        # Mock the chain construction: prompt | model | parser
        mock_prompt.__or__ = MagicMock()
        mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)
        
        raw_text = """
        0:01 welcome to today's lecture
        0:15 we will be discussing machine learning
        """
        
        with patch('src.refinery.cleaning._get_cleaning_prompt', return_value=mock_prompt), \
             patch('src.refinery.cleaning.StrOutputParser', return_value=mock_parser):
            
            result = clean_transcript_with_llm(raw_text)
            
            assert result == "This is a cleaned transcript with proper punctuation and formatting."
            mock_chain.invoke.assert_called_once()
    
    @patch('src.refinery.cleaning.model', None)  # Model not configured
    def test_llm_not_configured_fallback(self):
        """Test fallback to local cleaning when LLM is not configured."""
        raw_text = """
        0:01 welcome to today's lecture
        0:15 we will be discussing machine learning
        """
        
        with patch('src.refinery.cleaning._clean_transcript_locally') as mock_local:
            mock_local.return_value = "Locally cleaned transcript."
            
            result = clean_transcript_with_llm(raw_text)
            
            assert result == "Locally cleaned transcript."
            mock_local.assert_called_once_with(raw_text)
    
    @patch('src.refinery.cleaning.model')
    def test_llm_exception_fallback(self, mock_model):
        """Test fallback to local cleaning when LLM raises exception."""
        # Mock the LLM to raise an exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API Error")
        mock_model.__or__ = MagicMock(return_value=mock_chain)
        
        raw_text = """
        0:01 welcome to today's lecture
        0:15 we will be discussing machine learning
        """
        
        with patch('src.refinery.cleaning._get_cleaning_prompt') as mock_prompt, \
             patch('src.refinery.cleaning.StrOutputParser') as mock_parser, \
             patch('src.refinery.cleaning._clean_transcript_locally') as mock_local:
            
            mock_prompt.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            mock_local.return_value = "Fallback cleaned transcript."
            
            result = clean_transcript_with_llm(raw_text)
            
            assert result == "Fallback cleaned transcript."
            mock_local.assert_called_once_with(raw_text)
    
    def test_empty_text_handling(self):
        """Test handling of empty or minimal text."""
        # Empty text
        result = clean_transcript_with_llm("")
        assert result == ""
        
        # Very short text
        result = clean_transcript_with_llm("hi")
        assert result == ""
        
        # Text just under threshold
        short_text = "a" * 49  # Just under 50 character threshold
        result = clean_transcript_with_llm(short_text)
        assert result == ""
    
    @patch('src.refinery.cleaning.model')
    def test_text_length_threshold(self, mock_model):
        """Test that text must meet minimum length threshold."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Cleaned text"
        mock_model.__or__ = MagicMock(return_value=mock_chain)
        
        # Text that meets threshold
        long_enough_text = "a" * 60  # Above 50 character threshold
        
        with patch('src.refinery.cleaning._get_cleaning_prompt') as mock_prompt, \
             patch('src.refinery.cleaning.StrOutputParser') as mock_parser:
            
            mock_prompt.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            
            result = clean_transcript_with_llm(long_enough_text)
            
            assert result == "Cleaned text"
            mock_chain.invoke.assert_called_once()
    
    @patch('src.refinery.cleaning.model')
    def test_llm_response_stripping(self, mock_model):
        """Test that LLM response is properly stripped of whitespace."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "  \n  Cleaned transcript with extra whitespace  \n  "
        mock_model.__or__ = MagicMock(return_value=mock_chain)
        
        raw_text = "0:01 test transcript content for cleaning"
        
        with patch('src.refinery.cleaning._get_cleaning_prompt') as mock_prompt, \
             patch('src.refinery.cleaning.StrOutputParser') as mock_parser:
            
            mock_prompt.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            
            result = clean_transcript_with_llm(raw_text)
            
            assert result == "Cleaned transcript with extra whitespace"
    
    @patch('src.refinery.cleaning.model')
    def test_logging_behavior(self, mock_model, caplog):
        """Test that appropriate log messages are generated."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Cleaned transcript"
        mock_model.__or__ = MagicMock(return_value=mock_chain)
        
        raw_text = "0:01 test transcript for logging verification"
        
        with patch('src.refinery.cleaning._get_cleaning_prompt') as mock_prompt, \
             patch('src.refinery.cleaning.StrOutputParser') as mock_parser:
            
            mock_prompt.return_value = MagicMock()
            mock_parser.return_value = MagicMock()
            
            with caplog.at_level(logging.INFO):
                result = clean_transcript_with_llm(raw_text)
            
            # Should log the character count and LLM usage
            assert any("characters to LLM" in record.message for record in caplog.records)
    
    @patch('src.refinery.cleaning.model')
    def test_chain_construction(self, mock_model):
        """Test that the LangChain chain is constructed correctly."""
        mock_prompt = MagicMock()
        mock_parser = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Cleaned result"
        
        # Mock the chain construction (prompt | model | parser)
        mock_prompt.__or__ = MagicMock(return_value=MagicMock())
        mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)
        
        raw_text = "0:01 test transcript for chain construction test"
        
        with patch('src.refinery.cleaning._get_cleaning_prompt', return_value=mock_prompt), \
             patch('src.refinery.cleaning.StrOutputParser', return_value=mock_parser):
            
            result = clean_transcript_with_llm(raw_text)
            
            # Verify chain was invoked with correct parameters
            mock_chain.invoke.assert_called_once_with({"raw_text": raw_text})
            assert result == "Cleaned result"


class TestCleaningIntegration:
    """Integration tests combining multiple cleaning functions."""
    
    def test_llm_and_local_cleaning_consistency(self):
        """Test that LLM and local cleaning produce reasonable results."""
        raw_transcript = """
        0:01 welcome everyone to today's lecture
        0:30 we will be covering machine learning basics
        1:15 machine learning is important for ai
        2:00 lets start with definitions
        """
        
        # Test local cleaning
        local_result = _clean_transcript_locally(raw_transcript)
        
        # Verify local cleaning removes timestamps and adds structure
        assert "0:01" not in local_result
        assert "0:30" not in local_result
        assert "welcome everyone" in local_result.lower()
        assert "machine learning" in local_result.lower()
        
        # Test that result has proper sentence structure
        sentences = [s.strip() for s in local_result.split('.') if s.strip()]
        for sentence in sentences:
            if sentence:
                # Should start with capital letter
                assert sentence[0].isupper()
    
    @patch('src.refinery.cleaning.model')
    def test_error_recovery_chain(self, mock_model, caplog):
        """Test the complete error recovery chain from LLM to local fallback."""
        # First attempt: LLM fails with connection error
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = ConnectionError("Network error")
        
        # Create proper chain mock
        mock_prompt = MagicMock()
        mock_parser = MagicMock()
        mock_prompt.__or__ = MagicMock()
        mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)
        
        raw_text = """
        0:01 this is a test transcript
        0:30 with some timestamps to remove
        1:00 and content to clean up
        """
        
        with patch('src.refinery.cleaning._get_cleaning_prompt', return_value=mock_prompt), \
             patch('src.refinery.cleaning.StrOutputParser', return_value=mock_parser), \
             caplog.at_level(logging.WARNING):
            
            result = clean_transcript_with_llm(raw_text)
            
            # Should fallback to local cleaning
            assert "0:01" not in result  # Timestamps should be removed
            assert "0:30" not in result
            assert "1:00" not in result
            assert "test transcript" in result.lower()
            
            # Should log the fallback
            assert any("fallback" in record.message.lower() for record in caplog.records)
    
    def test_comprehensive_transcript_cleaning(self):
        """Test cleaning of a comprehensive, realistic transcript."""
        realistic_transcript = """
        0:00
        Welcome everyone
        0:15 today we're going to talk about
        0:30 machine learning and its applications
        1:00 um
        1:05 so machine learning is basically
        1:20 a way for computers to learn patterns
        1:45 without being explicitly programmed
        2:00 ok
        2:05 lets look at some examples
        2:30 in image recognition we can
        2:45 train models to identify objects
        3:00 like cats dogs cars etc
        """
        
        result = _clean_transcript_locally(realistic_transcript)
        
        # Verify comprehensive cleaning
        assert "0:00" not in result and "0:15" not in result and "0:30" not in result
        assert "1:00" not in result and "1:05" not in result and "1:20" not in result
        
        # Should preserve meaningful content
        assert "Welcome everyone" in result
        assert "machine learning" in result.lower()
        assert "image recognition" in result.lower()
        assert "cats dogs cars" in result.lower()
        
        # Should handle filler words appropriately
        # "um" and "ok" might create paragraph breaks
        paragraphs = result.split('\n\n')
        assert len(paragraphs) >= 2  # Should create multiple paragraphs
        
        # Should add proper punctuation
        assert result.strip().endswith('.')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])