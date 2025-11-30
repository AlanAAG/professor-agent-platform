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


def _clean_source_tag(text: str) -> str:
    """Helper to strip artifacts like <v Speaker> from strings."""
    # Remove <...> tags
    return re.sub(r"<[^>]+>", "", text).strip()


def _split_into_lecture_segments(full_text: str, filename: str) -> List[Dict[str, Any]]:
    """
    Split a full multi-lecture transcript text into individual lecture segments.

    Refactored Strategy: "Date-Anchor"
    1. Find all lines starting with a date pattern (DD/MM/YYYY).
    2. The "Title Block" is everything before the date line (up to a blank line or start of file).
    3. The "Teacher Name" is the line immediately following the date, unless it looks like a timestamp/body.
    4. The Body is the rest until the next segment starts.
    """
    segments: List[Dict[str, Any]] = []
    class_name = os.path.basename(filename)
    if class_name.lower().endswith(".txt"):
        class_name = class_name[:-4]

    lines = full_text.splitlines()

    # Regex to identify the date anchor line (DD/MM/YYYY)
    # Using \d{1,2}/\d{1,2}/\d{4} to match dates like 20/05/2023 or 1/1/2024
    # We look for this pattern at the start of a line (ignoring whitespace)
    date_anchor_pattern = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}")

    # Find all line indices that match the date pattern
    date_indices = [i for i, line in enumerate(lines) if date_anchor_pattern.match(line)]

    if not date_indices:
        logging.warning(f"No date anchors found in {filename} with pattern DD/MM/YYYY.")
        return []

    # Identify segment boundaries
    # Each segment is defined by its Date Anchor.
    # Title is above Date. Body is below Date (and optional Teacher).

    segment_infos = []

    for i, date_idx in enumerate(date_indices):
        # 1. Capture Title Block
        # Walk backwards from date_idx - 1 until we hit a blank line or start of file
        title_lines = []
        curr = date_idx - 1
        while curr >= 0 and lines[curr].strip() != "":
            title_lines.append(lines[curr])
            curr -= 1

        # The lines were collected in reverse order (closest to date first)
        # We need to reverse them back to original order
        title_lines.reverse()

        # Clean tags from title lines
        cleaned_title_lines = [_clean_source_tag(l) for l in title_lines]
        title = " - ".join(cleaned_title_lines)
        if not title:
             title = "Untitled Segment"

        # 2. Capture Date
        date_line = lines[date_idx].strip()
        lecture_date = parse_general_date(date_line)

        # 3. Capture Teacher Name (Optional) and determine Body Start
        teacher_name = "Unknown Instructor"
        body_start_idx = date_idx + 1

        if body_start_idx < len(lines):
            potential_teacher_line = lines[body_start_idx]
            # Check if it looks like a timestamp (starts with a number)
            # e.g., "0:00 Welcome" or "10:00 ..."
            if re.match(r"^\s*\d", potential_teacher_line):
                # It looks like body content (timestamp), so teacher is missing
                pass
            elif potential_teacher_line.strip() == "":
                 # Empty line? Skip it, assume teacher is missing.
                 # Body starts after this blank line if next line is body?
                 # If empty, we just skip checking for teacher name and use default.
                 # Body starts at the line after.
                 pass
            else:
                 # It's a non-empty line not starting with a number -> Teacher Name
                 teacher_name = _clean_source_tag(potential_teacher_line)
                 body_start_idx += 1

        # 4. Determine Body End
        # The body goes until the start of the *next* title block.
        # We need to know where the next title block starts.
        # The next title block starts at `next_date_idx - (number of title lines)`.
        # Actually, we calculated title by walking back from date until blank line.
        # So the *next* segment starts at the blank line before its title.

        if i < len(date_indices) - 1:
            next_date_idx = date_indices[i+1]
            # Find start of next title
            curr_next = next_date_idx - 1
            while curr_next >= 0 and lines[curr_next].strip() != "":
                curr_next -= 1
            # curr_next is now the index of the blank line (or -1)
            # So body of current segment ends at curr_next (exclusive)
            body_end_idx = curr_next
            # Handle edge case where body_start_idx > body_end_idx (overlapping/messy)
            if body_end_idx < body_start_idx:
                body_end_idx = body_start_idx # Empty body
        else:
            body_end_idx = len(lines)

        raw_body_lines = lines[body_start_idx:body_end_idx]
        body = "\n".join(raw_body_lines).strip()

        # Validation
        if len(body) < 50:
            logging.warning(
                f"Skipping segment in {filename}: transcript body too short (<50 chars). "
                f"Title: {title}"
            )
            continue

        segments.append({
            "class_name": class_name,
            "title": title,
            "lecture_date": lecture_date,
            "teacher_name": teacher_name,
            "transcript_body": body
        })

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
        # Teacher Name is now optional (defaults to Unknown Instructor), so checking for key existence is enough
        if 'teacher_name' not in seg:
             logging.error("Missing teacher name key")
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
