# src/shared/utils.py

import re
import os
import logging
from typing import List, Any

try:
    import cohere
except Exception:
    cohere = None  # Optional dependency

# Shared constants
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "models/embedding-001")


def cohere_rerank(query: str, documents: List[Any]) -> List[Any]:
    """Re-rank a list of documents using Cohere, if configured.

    - Accepts either list of strings or objects with 'page_content'/'content'.
    - Returns list reordered by relevance. If Cohere is not configured or
      an error occurs, returns the input order.
    """
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key or not cohere or not documents:
        return documents

    try:
        client = cohere.Client(api_key)
        # Normalize to strings list and keep mapping
        texts: List[str] = []
        index_to_doc: dict[int, Any] = {}
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                content = doc
            elif isinstance(doc, dict):
                content = doc.get("page_content") or doc.get("content") or ""
            else:
                content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or ""
            texts.append(content)
            index_to_doc[i] = doc

        results = client.rerank(
            query=query,
            documents=texts,
            top_n=len(texts),
            model=os.environ.get("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
        )
        return [index_to_doc[r.index] for r in results.results]
    except Exception as e:
        logging.warning(f"Cohere re-ranking failed, using original order: {e}")
        return documents
import datetime
import logging
from typing import Optional
from dateutil import parser # Use dateutil for flexible parsing
from dateutil import tz

# --- Setup Logging ---
# Ensure logging is configured if run directly or imported early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_session_date(date_str: str) -> Optional[datetime.datetime]:
    """
    Parses specific date strings like 'DD/MM/YYYY'.
    Returns a datetime object (UTC) or None if parsing fails.
    """
    # ... (Keep existing implementation) ...
    if not date_str: return None
    date_part = date_str.split('|')[0].strip().split(' ')[0].strip()
    try:
        dt_obj = datetime.datetime.strptime(date_part, '%d/%m/%Y')
        return dt_obj.replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        logging.warning(f"Could not parse date string: '{date_str}' with format DD/MM/YYYY")
        return None

# --- NEW: Flexible Date Parser ---
def parse_general_date(date_str: str, local_tz: tz.tzfile = tz.gettz('Asia/Dubai')) -> Optional[datetime.datetime]:
    """
    Attempts to parse various common date formats using dateutil.parser.
    Returns a datetime object (UTC) or None if parsing fails.
    Handles potential extra text.
    """
    if not date_str:
        return None

    # Clean the string a bit, remove common extra info like time zones or day names if needed
    date_part = date_str.split('|')[0].strip() # Remove time if separated by |

    try:
        # dateutil.parser.parse is very flexible
        # dayfirst=True helps interpret DD/MM vs MM/DD correctly in ambiguous cases
        dt_obj = parser.parse(date_part, dayfirst=True)

        # If the parsed datetime is naive, assume local timezone (Dubai by default)
        if dt_obj.tzinfo is None:
            assumed_tz = local_tz or tz.gettz('Asia/Dubai')
            dt_obj = dt_obj.replace(tzinfo=assumed_tz)

        # Always convert to UTC for consistency
        return dt_obj.astimezone(datetime.timezone.utc)

    except (ValueError, OverflowError, TypeError) as e:
        logging.warning(f"Could not parse general date string: '{date_str}'. Error: {e}")
        # Optionally, try the specific session date parser as a fallback
        # return parse_session_date(date_str)
        return None

def create_safe_filename(text: str, max_length: int = 100) -> str:
    """
    Cleans a string to make it suitable for use as a filename.
    """
    # ... (Keep existing implementation) ...
    if not text: return "untitled"
    text = re.sub(r'[^\w\-]+', '_', text)
    text = re.sub(r'_+', '_', text)
    text = text.strip('_')
    return text[:max_length]

# --- Example Usage ---
if __name__ == "__main__":
    print("Testing utility functions...")
    # ... (Keep existing tests) ...

    # Test general date parsing
    print("\nTesting parse_general_date:")
    date_g1 = "22/10/2025"
    date_g2 = "October 22, 2025"
    date_g3 = "2025-10-22"
    date_g4 = "Invalid Date String"
    date_g5 = "22 Oct 2025 | 10:00 AM"
    print(f"'{date_g1}' -> {parse_general_date(date_g1)}")
    print(f"'{date_g2}' -> {parse_general_date(date_g2)}")
    print(f"'{date_g3}' -> {parse_general_date(date_g3)}")
    print(f"'{date_g4}' -> {parse_general_date(date_g4)}")
    print(f"'{date_g5}' -> {parse_general_date(date_g5)}")