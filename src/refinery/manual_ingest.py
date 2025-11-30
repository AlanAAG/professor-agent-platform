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

# --- Robust Segmentation Pattern ---
# 1. title_block: 1+ lines. MUST NOT start with a timestamp (digit:digit).
# 2. date_line: DD/MM/YYYY.
# 3. teacher_line: Optional. Must not start with number.
# 4. body: Everything else.
LECTURE_SEGMENTATION_PATTERN = re.compile(
    # Title: One or more lines, but NONE of them can start with a digit (prevent capturing timestamps)
    r"^\s*(?P<title_block>(?:(?!\d+:\d+)[^\n]+\n)+?)"
    # Date: The anchor (e.g. 24/09/2025)
    r"(?P<date_line>\d{2}/\d{2}/\d{4}.*?)\n"
    # Teacher: Optional line (if it doesn't start with a number)
    r"(?:(?P<teacher_line>[^0-9\n].*?)\n)?"
    # Body: The rest
    r"(?P<body>.*?)"
    # Stop at: Next Date Pattern (Lookahead) or End of String
    r"(?=\n\s*(?:(?!\d+:\d+)[^\n]+\n)+?\d{2}/\d{2}/\d{4}|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)

def _clean_source_tag(text: str) -> str:
    """Removes tags and trims whitespace."""
    if not text:
        return ""
    # Remove anywhere in the line
    text = re.sub(r"\", "", text)
    return text.strip()

def _split_into_lecture_segments(full_text: str, filename: str) -> List[Dict[str, Any]]:
    """
    Split a full multi-lecture transcript text into individual lecture segments.
    """
    segments: List[Dict[str, Any]] = []
    class_name = os.path.basename(filename)
    if class_name.lower().endswith(".txt"):
        class_name = class_name[:-4]

    # 1. GLOBAL CLEAN: Remove all tags first to fix segmentation
    clean_full_text = re.sub(r"\", "", full_text)

    for match in LECTURE_SEGMENTATION_PATTERN.finditer(clean_full_text):
        raw_title_block = match.group("title_block")
        date_str = match.group("date_line").strip()
        
        teacher_name = "Unknown Instructor"
        if match.group("teacher_line"):
            teacher_name = match.group("teacher_line").strip()
            
        body = match.group("body").strip()

        # Join title lines
        title_lines = [
            line.strip() 
            for line in raw_title_block.split('\n') 
            if line.strip()
        ]
        title = " - ".join(title_lines)

        lecture_date = parse_general_date(date_str)

        # Basic Validation
        if not title:
            logging.warning(f"Skipping segment in {filename}: empty title.")
            continue
        
        if len(body) < 50:
            logging.warning(f"Skipping segment in {filename}: body too short.")
            continue

        segments.append({
            "class_name": class_name,
            "title": title,
            "lecture_date": lecture_date,
            "teacher_name": teacher_name,
            "transcript_body": body,
        })

    return segments

def process_manual_transcript(file_path: str) -> Tuple[int, List[str]]:
    """
    Process a single manually-provided transcript file.
    """
    successful_segments = 0
    failed_titles: List[str] = []

    try:
        file_path = str(Path(file_path).resolve())
        filename = os.path.basename(file_path)
        logging.info(f"Processing manual transcript file: {filename}")

        if not filename.endswith('.txt'):
            return (0, [f"Invalid file type: {filename}"])

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        except Exception as e:
            return (0, [f"Read error: {filename}"])

        segments = _split_into_lecture_segments(full_text, filename)
        if not segments:
            return (0, [f"No segments: {filename}"])

        for segment in segments:
            segment_title = segment.get("title") or "Untitled"
            source_key = f"{filename}|{segment_title}"

            # Check DB existence (Sync)
            try:
                if url_exists_in_db_sync(source_key):
                    logging.info(f"Segment already exists (dedup): {source_key}")
                    successful_segments += 1
                    continue
            except Exception:
                pass

            raw_body = segment.get("transcript_body") or ""
            
            # Clean with LLM
            try:
                clean_text = clean_transcript_with_llm(raw_body)
                if not clean_text:
                    failed_titles.append(segment_title)
                    continue
            except Exception:
                failed_titles.append(segment_title)
                continue

            embedding_metadata: Dict[str, Any] = {
                "class_name": segment["class_name"],
                "content_type": "manual_transcript",
                "title": segment_title,
                "lecture_date": segment["lecture_date"].isoformat() if segment.get("lecture_date") else None,
                "teacher_name": segment.get("teacher_name"),
                "source_file": filename,
                "source_url": source_key,
                "retrieval_date": datetime.datetime.now().isoformat(),
            }

            try:
                chunk_and_embed_text(clean_text, embedding_metadata)
                logging.info(f"✅ Embedded segment '{segment_title}'")
                successful_segments += 1
            except Exception:
                failed_titles.append(segment_title)

        return (successful_segments, failed_titles)

    except Exception as e:
        logging.error(f"Unexpected error processing {file_path}: {e}")
        return (successful_segments, failed_titles)

def ingest_all_manual_transcripts(directory: str = "data/raw_transcripts/manual/") -> Dict[str, Any]:
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return {"total": 0, "successful": 0, "failed": ["Directory not found"]}
    
    txt_files = list(directory_path.glob("*.txt"))
    logging.info(f"Found {len(txt_files)} .txt files")
    
    total_segments = 0
    successful_segments = 0
    failed_titles: List[str] = []
    
    for txt_file in txt_files:
        try:
            succ, failed = process_manual_transcript(str(txt_file))
            total_segments += succ + len(failed)
            successful_segments += succ
            failed_titles.extend(failed)
        except Exception as e:
            failed_titles.append(f"{txt_file.name} (error)")
    
    return {
        "total_segments": total_segments,
        "successful_segments": successful_segments,
        "failed_titles": failed_titles,
    }

# --- Validation Helper ---
def validate_file_format(file_path: str) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        segments = _split_into_lecture_segments(full_text, os.path.basename(file_path))
        if not segments:
            logging.error("No segments found.")
            return False
        for i, seg in enumerate(segments):
            logging.info(f"Segment {i+1}: Title='{seg['title']}', Date={seg['lecture_date']}, Teacher='{seg['teacher_name']}'")
        return True
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_all_manual_transcripts()