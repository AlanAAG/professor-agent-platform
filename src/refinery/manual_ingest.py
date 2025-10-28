# src/refinery/manual_ingest.py

import os
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.refinery.cleaning import clean_transcript_with_llm
from src.refinery.embedding import chunk_and_embed_text, url_exists_in_db_sync
from src.shared.utils import parse_general_date


def _extract_metadata_from_file(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from the first 3 lines of a manual transcript file.
    
    Expected format:
    Line 1: {ClassName} - {Lecture Title}
    Line 2: Date: {Date} | Time: {Time}
    Line 3: (empty line)
    
    Args:
        file_path: Path to the transcript file
        
    Returns:
        Dict containing extracted metadata
        
    Raises:
        ValueError: If file format doesn't match expected structure
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 3:
            raise ValueError(f"File must have at least 3 lines, found {len(lines)}")
        
        # Parse line 1: Subject - Lecture Title
        line1 = lines[0].strip()
        if ' - ' not in line1:
            raise ValueError(f"Line 1 must contain ' - ' separator, found: '{line1}'")
        
        class_name, title = line1.split(' - ', 1)
        class_name = class_name.strip()
        title = title.strip()
        
        # Parse line 2: Date: YYYY-MM-DD | Time: HH:MM AM/PM
        line2 = lines[1].strip()
        if not line2.startswith('Date: '):
            raise ValueError(f"Line 2 must start with 'Date: ', found: '{line2}'")
        
        date_time_str = line2[6:]  # Remove 'Date: ' prefix
        
        # Parse date using flexible parser
        lecture_date = parse_general_date(date_time_str)
        
        return {
            'class_name': class_name,
            'title': title,
            'lecture_date': lecture_date,
            'raw_date_string': date_time_str
        }
        
    except Exception as e:
        logging.error(f"Error extracting metadata from {file_path}: {e}")
        raise


def _extract_transcript_body(file_path: str) -> str:
    """
    Extract the transcript content starting from line 4.
    
    Args:
        file_path: Path to the transcript file
        
    Returns:
        String containing the full transcript body
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 4:
            logging.warning(f"File {file_path} has fewer than 4 lines, using all content after line 3")
            return ""
        
        # Join all lines from line 4 onwards (index 3)
        transcript_body = ''.join(lines[3:])
        return transcript_body.strip()
        
    except Exception as e:
        logging.error(f"Error reading transcript body from {file_path}: {e}")
        raise


def process_manual_transcript(file_path: str) -> bool:
    """
    Process a single manually-provided transcript file.
    
    Steps:
    1. Parse filename to extract class_name (e.g., "Statistics.txt" -> "Statistics")
    2. Read file and extract metadata from first 3 lines
    3. Parse date/time from line 2 using shared utils
    4. Extract full transcript body (line 4 onwards)
    5. Clean transcript using cleaning.clean_transcript_with_llm()
    6. Embed using embedding.chunk_and_embed_text() with metadata:
       {
           "class_name": ,
           "content_type": "manual_transcript",
           "title": ,
           "lecture_date": ,
           "source_file": ,
           "retrieval_date": 
       }
    7. Return True on success, log and return False on failure
    
    Args:
        file_path: Full path to .txt file
        
    Returns:
        bool: Success status
    """
    try:
        file_path = str(Path(file_path).resolve())
        filename = os.path.basename(file_path)
        
        logging.info(f"Processing manual transcript: {filename}")
        
        # Step 1: Extract class name from filename
        if not filename.endswith('.txt'):
            logging.warning(f"Skipping non-txt file: {filename}")
            return False
        
        class_name_from_file = filename[:-4]  # Remove .txt extension
        
        # Step 2: Extract metadata from first 3 lines
        try:
            metadata = _extract_metadata_from_file(file_path)
            logging.info(f"Extracted metadata - Class: {metadata['class_name']}, Title: {metadata['title']}")
        except ValueError as e:
            logging.error(f"Invalid file format for {filename}: {e}")
            return False
        
        # Validate class name consistency
        if metadata['class_name'] != class_name_from_file:
            logging.warning(f"Class name mismatch: filename='{class_name_from_file}' vs content='{metadata['class_name']}'. Using content.")
        
        # Step 3: Check for de-duplication
        source_file_key = filename
        if url_exists_in_db_sync(source_file_key):
            logging.info(f"File {filename} already exists in database. Skipping.")
            return True  # Consider this a success since it's already processed
        
        # Step 4: Extract transcript body
        try:
            transcript_body = _extract_transcript_body(file_path)
            if not transcript_body:
                logging.warning(f"No transcript content found in {filename}")
                return False
            
            logging.info(f"Extracted {len(transcript_body)} characters of transcript content")
        except Exception as e:
            logging.error(f"Failed to extract transcript body from {filename}: {e}")
            return False
        
        # Step 5: Clean transcript using LLM
        try:
            clean_text = clean_transcript_with_llm(transcript_body)
            if not clean_text:
                logging.warning(f"Transcript cleaning produced empty result for {filename}")
                return False
            
            logging.info(f"Cleaned transcript: {len(clean_text)} characters")
        except Exception as e:
            logging.error(f"Failed to clean transcript for {filename}: {e}")
            return False
        
        # Step 6: Prepare metadata for embedding
        embedding_metadata = {
            "class_name": metadata['class_name'],
            "content_type": "manual_transcript",
            "title": metadata['title'],
            "lecture_date": metadata['lecture_date'].isoformat() if metadata['lecture_date'] else None,
            "source_file": source_file_key,
            "retrieval_date": datetime.datetime.now().isoformat()
        }
        
        # Step 7: Embed the cleaned text
        try:
            chunk_and_embed_text(clean_text, embedding_metadata)
            logging.info(f"✅ Successfully embedded {filename}")
            return True
        except Exception as e:
            logging.error(f"Failed to embed transcript for {filename}: {e}")
            return False
            
    except Exception as e:
        logging.error(f"Unexpected error processing {file_path}: {e}")
        return False


def ingest_all_manual_transcripts(directory: str = "data/raw_transcripts/manual/") -> Dict[str, Any]:
    """
    Process all .txt files in the manual transcripts directory.
    
    Args:
        directory: Directory containing manual transcript files
        
    Returns:
        dict: Statistics {
            "total": int,
            "successful": int,
            "failed": list[str],  # filenames that failed
        }
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logging.error(f"Directory does not exist: {directory}")
        return {
            "total": 0,
            "successful": 0,
            "failed": [f"Directory not found: {directory}"]
        }
    
    if not directory_path.is_dir():
        logging.error(f"Path is not a directory: {directory}")
        return {
            "total": 0,
            "successful": 0,
            "failed": [f"Not a directory: {directory}"]
        }
    
    # Find all .txt files
    txt_files = list(directory_path.glob("*.txt"))
    
    if not txt_files:
        logging.warning(f"No .txt files found in {directory}")
        return {
            "total": 0,
            "successful": 0,
            "failed": []
        }
    
    logging.info(f"Found {len(txt_files)} .txt files to process")
    
    # Process each file
    successful = 0
    failed = []
    
    for txt_file in txt_files:
        try:
            if process_manual_transcript(str(txt_file)):
                successful += 1
            else:
                failed.append(txt_file.name)
        except Exception as e:
            logging.error(f"Critical error processing {txt_file.name}: {e}")
            failed.append(txt_file.name)
    
    # Return statistics
    stats = {
        "total": len(txt_files),
        "successful": successful,
        "failed": failed
    }
    
    logging.info(f"Processing complete: {successful}/{len(txt_files)} successful")
    if failed:
        logging.warning(f"Failed files: {failed}")
    
    return stats


# --- Testing/Debug Functions ---
def validate_file_format(file_path: str) -> bool:
    """
    Validate that a file follows the expected manual transcript format.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if format is valid, False otherwise
    """
    try:
        metadata = _extract_metadata_from_file(file_path)
        transcript_body = _extract_transcript_body(file_path)
        
        if not metadata['class_name']:
            logging.error("Missing class name")
            return False
        
        if not metadata['title']:
            logging.error("Missing title")
            return False
        
        if not transcript_body:
            logging.error("Missing transcript body")
            return False
        
        logging.info(f"✅ File format validation passed for {os.path.basename(file_path)}")
        logging.info(f"   Class: {metadata['class_name']}")
        logging.info(f"   Title: {metadata['title']}")
        logging.info(f"   Date: {metadata['lecture_date']}")
        logging.info(f"   Content length: {len(transcript_body)} characters")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ File format validation failed: {e}")
        return False


if __name__ == "__main__":
    # Basic test when run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_dir = "data/raw_transcripts/manual/"
    logging.info(f"Testing manual ingestion from {test_dir}")
    
    if os.path.exists(test_dir):
        stats = ingest_all_manual_transcripts(test_dir)
        logging.info(f"Test results: {stats}")
    else:
        logging.info(f"Test directory {test_dir} does not exist")