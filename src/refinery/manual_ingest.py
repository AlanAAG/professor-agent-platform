# src/refinery/manual_ingest.py

import os
import re
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from src.refinery.cleaning import clean_transcript_with_llm
from src.refinery.embedding import chunk_and_embed_text, url_exists_in_db_sync
from src.shared.utils import parse_general_date


# --- New multi-lecture segmentation ---
# Pattern captures segments with 3-line header:
# 1) title (non-empty line)
# 2) date/time (flexible, parsed later)
# 3) teacher name (non-empty)
# 4+) transcript body until a blank line followed by the next header, or EOF
# Notes:
# - Use non-greedy body capture with DOTALL
# - Lookahead asserts either two newlines then 3 non-empty header lines, or end of file
LECTURE_SEGMENTATION_PATTERN = re.compile(
    r"^\s*(?P<title>[^\n].*?)\n"  # Line 1: title
    r"(?P<date_time>[^\n].*?)\n"     # Line 2: date time
    r"(?P<teacher_name>[^\n].*?)\n"  # Line 3: teacher name
    r"(?P<body>.*?)(?=\n{2}(?=[^\n]+\n[^\n]+\n[^\n]+\n)|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)


def _split_into_lecture_segments(full_text: str, filename: str) -> List[Dict[str, Any]]:
    """
    Split a full multi-lecture transcript text into individual lecture segments.

    New file format per lecture:
    Line 1: {Lecture Title}
    Line 2: {Date of Lecture} {Time of Class}
    Line 3: {Teacher's Name}
    Line 4+: Transcript body until one blank line ("\n\n") before the next header or EOF.

    class_name is derived from the filename (e.g., "Statistics.txt" -> "Statistics").
    """
    segments: List[Dict[str, Any]] = []
    class_name = os.path.basename(filename)
    if class_name.lower().endswith(".txt"):
        class_name = class_name[:-4]

    for match in LECTURE_SEGMENTATION_PATTERN.finditer(full_text or ""):
        title = (match.group("title") or "").strip()
        date_time_str = (match.group("date_time") or "").strip()
        teacher_name = (match.group("teacher_name") or "").strip()
        body = (match.group("body") or "").strip()

        lecture_date = parse_general_date(date_time_str)

        segments.append(
            {
                "class_name": class_name,
                "title": title,
                "lecture_date": lecture_date,
                "teacher_name": teacher_name,
                "transcript_body": body,
            }
        )

    return segments


def process_manual_transcript(file_path: str) -> Tuple[int, List[str]]:
    """
    Process a single manually-provided transcript file that may contain multiple lectures.
    
    Steps:
    1. Read entire file content
    2. Split into lecture segments using regex
    3. For each segment: clean, de-duplicate (by filename|title), and embed
    4. Return (successful_segment_count, [failed_segment_titles])
    
    Args:
        file_path: Full path to .txt file
        
    Returns:
        Tuple[int, List[str]]: (count of successful segments, list of failed titles)
    """
    successful_segments = 0
    failed_titles: List[str] = []

    try:
        file_path = str(Path(file_path).resolve())
        filename = os.path.basename(file_path)

        logging.info(f"Processing manual transcript file (multi-lecture): {filename}")

        if not filename.endswith('.txt'):
            logging.warning(f"Skipping non-txt file: {filename}")
            return (0, [f"Invalid file type: {filename}"])

        # Read full file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except Exception as e:
            logging.error(f"Failed to read file {filename}: {e}")
            return (0, [f"Read error: {filename}"])

        # Split into segments
        segments = _split_into_lecture_segments(full_text, filename)
        if not segments:
            logging.warning(f"No lecture segments detected in {filename}")
            return (0, [f"No segments: {filename}"])

        # Process each segment
        for segment in segments:
            segment_title = segment.get("title") or "Untitled"
            source_key = f"{filename}|{segment_title}"

            # De-duplication by unique segment key
            try:
                if url_exists_in_db_sync(source_key):
                    logging.info(f"Segment already exists (dedup): {source_key} -> skipping")
                    successful_segments += 1  # Consider dedup as success
                    continue
            except Exception as e:
                # Fail-open: do not treat check failure as hard failure for the segment
                logging.warning(f"Dedup check failed for {source_key}: {e}")

            raw_body = segment.get("transcript_body") or ""
            if not raw_body.strip():
                logging.warning(f"Empty transcript body for segment '{segment_title}' in {filename}")
                failed_titles.append(segment_title)
                continue

            # Clean transcript
            try:
                clean_text = clean_transcript_with_llm(raw_body)
                if not clean_text:
                    logging.warning(f"Cleaning produced empty result for segment '{segment_title}' in {filename}")
                    failed_titles.append(segment_title)
                    continue
            except Exception as e:
                logging.error(f"Failed cleaning for segment '{segment_title}' in {filename}: {e}")
                failed_titles.append(segment_title)
                continue

            # Prepare embedding metadata (include new teacher_name and dedupe key as source_url)
            embedding_metadata: Dict[str, Any] = {
                "class_name": segment["class_name"],
                "content_type": "manual_transcript",
                "title": segment["title"],
                "lecture_date": segment["lecture_date"].isoformat() if segment.get("lecture_date") else None,
                "teacher_name": segment.get("teacher_name"),
                "source_file": filename,
                "source_url": source_key,
                "retrieval_date": datetime.datetime.now().isoformat(),
            }

            # Embed
            try:
                chunk_and_embed_text(clean_text, embedding_metadata)
                logging.info(f"✅ Embedded segment '{segment_title}' from {filename}")
                successful_segments += 1
            except Exception as e:
                logging.error(f"Failed to embed segment '{segment_title}' in {filename}: {e}")
                failed_titles.append(segment_title)

        return (successful_segments, failed_titles)

    except Exception as e:
        logging.error(f"Unexpected error processing {file_path}: {e}")
        return (successful_segments, failed_titles)


def ingest_all_manual_transcripts(directory: str = "data/raw_transcripts/manual/") -> Dict[str, Any]:
    """
    Process all .txt files in the manual transcripts directory.
    
    Args:
        directory: Directory containing manual transcript files
        
    Returns:
        dict: Segment-level statistics {
            "total_segments": int,
            "successful_segments": int,
            "failed_titles": list[str],  # lecture titles that failed
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
    
    # Process each file and aggregate segment stats
    total_segments = 0
    successful_segments = 0
    failed_titles: List[str] = []
    
    for txt_file in txt_files:
        try:
            succ_count, file_failed_titles = process_manual_transcript(str(txt_file))
            # Total is successes + failures (dedup counted as success inside)
            total_segments += succ_count + len(file_failed_titles)
            successful_segments += succ_count
            failed_titles.extend(file_failed_titles)
        except Exception as e:
            logging.error(f"Critical error processing {txt_file.name}: {e}")
            # Cannot determine how many segments; count as 0 success, 0 total increment, but note filename
            failed_titles.append(f"{txt_file.name} (file error)")
    
    # Return statistics
    stats = {
        "total_segments": total_segments,
        "successful_segments": successful_segments,
        "failed_titles": failed_titles,
    }
    
    logging.info(
        f"Processing complete: {successful_segments}/{total_segments} segments successful across {len(txt_files)} files"
    )
    if failed_titles:
        logging.warning(f"Failed titles: {failed_titles}")
    
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
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        filename = os.path.basename(file_path)
        segments = _split_into_lecture_segments(full_text, filename)
        if not segments:
            logging.error("No lecture segments found")
            return False

        # Basic sanity checks on first segment
        seg = segments[0]
        if not seg.get('class_name'):
            logging.error("Missing class name (from filename)")
            return False
        if not seg.get('title'):
            logging.error("Missing lecture title")
            return False
        if not seg.get('teacher_name'):
            logging.error("Missing teacher name")
            return False
        if not seg.get('transcript_body'):
            logging.error("Missing transcript body")
            return False

        logging.info(f"✅ File format validation passed for {os.path.basename(file_path)} with {len(segments)} segment(s)")
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