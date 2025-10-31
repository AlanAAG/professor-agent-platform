# src/refinery/embedding.py

import os
import logging
import datetime
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from supabase.client import Client, create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.shared.utils import EMBEDDING_MODEL_NAME
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# --- Load Environment Variables ---
# Ensures .env file is read when running locally.
load_dotenv()

# --- 1. Initialize Clients ---
supabase = None
embeddings_model = None
vector_store = None # Initialize vector_store as None initially

try:
    # --- Check and Load Environment Variables ---
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set in .env for embeddings")

    # --- Initialize Supabase Client ---
    supabase = create_client(supabase_url, supabase_key)
    logging.info("Supabase client initialized.")

    # --- Initialize Embedding Model ---
    # Keep model name in one shared place for consistency
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=gemini_api_key
    )
    logging.info("Gemini embedding model initialized.")

    # --- Initialize the Vector Store Client ---
    # Moved initialization here after ensuring supabase and embeddings_model are valid
    vector_store = SupabaseVectorStore(
        client=supabase,
        table_name="documents", # The table you created in Supabase SQL Editor
        embedding=embeddings_model
    )
    logging.info("SupabaseVectorStore initialized.")

    logging.info("✅ Embedding clients configured successfully.")

except Exception as e:
    # --- Improved Error Handling ---
    logging.error("Could not initialize embedding clients.")
    logging.error(f"Error details: {e}")
    logging.error("Please double-check your .env file for correct SUPABASE_URL, SUPABASE_KEY, and GEMINI_API_KEY.")
    # Do not exit here; allow importing modules that handle missing clients gracefully.

# --- 3. Initialize the Text Splitter ---
# This object is responsible for "chunking" your large text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, # Max characters per chunk (adjust if needed)
    chunk_overlap=200, # Characters to overlap between chunks (helps maintain context)
    length_function=len
)
logging.info("Text splitter initialized.")


# --- Metadata Validation ---
REQUIRED_METADATA = ["class_name", "content_type"]
OPTIONAL_METADATA = [
    "source_file",
    "source_url",
    "title",
    "lecture_date",
    "page_number",
    "links",
    "retrieval_date",
    "content_hash",
]

def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and normalizes metadata before embedding.

    Ensures required fields exist, adds a default retrieval_date if missing,
    and removes None-valued entries.
    """
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary.")

    for field in REQUIRED_METADATA:
        if field not in metadata:
            raise ValueError(f"Missing required metadata field: {field}")

    if "retrieval_date" not in metadata:
        metadata["retrieval_date"] = datetime.datetime.now().isoformat()

    if "content_hash" in metadata and metadata["content_hash"] is not None:
        metadata["content_hash"] = str(metadata["content_hash"])

    # Remove None values
    return {k: v for k, v in metadata.items() if v is not None}


# --- Retry wrapper for vector store writes ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def _add_documents_with_retry(documents: List[Any]):
    if not vector_store:
        raise EnvironmentError("Vector store is not initialized.")
    return vector_store.add_documents(documents)


# --- Main Function to be Called by Other Scripts ---
def chunk_and_embed_text(clean_text: str, metadata: Dict[str, Any]):
    """
    Takes clean text, splits it into chunks, and embeds it
    into the Supabase vector store with metadata.
    """
    if not vector_store:
        # This check is important because initialization may have failed gracefully.
        raise EnvironmentError("Vector store is not initialized.")
    if not clean_text:
        logging.info("-> Skipping embedding: No clean text provided.")
        return

    logging.info(f"-> Chunking and embedding for: {metadata.get('class_name')}")

    # 1. Split the clean text into LangChain "Document" objects (chunks)
    documents = text_splitter.create_documents([clean_text])
    logging.info(f"   Split text into {len(documents)} chunks.")

    # 2. Validate and attach metadata to each chunk
    normalized_metadata = validate_metadata(metadata)
    for doc in documents:
        doc.metadata = normalized_metadata

    # 3. Add all prepared documents to Supabase
    # LangChain's SupabaseVectorStore handles calculating embeddings
    # and saving everything to your 'documents' table.
    try:
        _add_documents_with_retry(documents)
        logging.info(f"✅ Embedding successful for {len(documents)} chunks.")
    except Exception as e:
        logging.error(f"❌ Failed after retries during Supabase add_documents: {e}")
        # Re-raising ensures the calling script knows about the failure.
        raise


def check_if_embedded(filter: Dict[str, Any]) -> bool:
    """
    (Advanced) Checks if a document (like a syllabus) with specific metadata
    already exists in the database.
    """
    # For now, just return False so it always embeds new static files.
    # You can implement this later using vector_store.similarity_search()
    # or direct Supabase queries if needed.
    return False


async def check_if_embedded_recently(filter: Dict[str, Any], days: int = 7) -> bool:
    """Checks Supabase metadata to see if content matching the filter was processed recently."""
    if not supabase:
        # Default to re-processing if Supabase is not configured
        return False

    if not isinstance(filter, dict):
        logging.warning("check_if_embedded_recently called with invalid filter (expected dict).")
        return False

    try:
        # Use naive ISO timestamps to match how metadata['retrieval_date'] is stored
        cutoff_dt = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_iso = cutoff_dt.isoformat()

        def _execute_query():
            # Base query: restrict by recent retrieval_date
            query = supabase.table("documents").select("metadata", count="exact")

            # Prefer strict URL identity when available
            if "source_url" in filter and filter["source_url"]:
                query = query.eq("metadata->>source_url", filter["source_url"])
            elif filter:
                # Fall back to JSON contains on metadata for provided keys
                query = query.contains("metadata", filter)

            query = query.gt("metadata->>retrieval_date", cutoff_iso).limit(1)
            return query.execute()

        response = await asyncio.to_thread(_execute_query)

        # supabase-py may return an object with attribute or dict with 'count'
        count = getattr(response, "count", None)
        if count is None and isinstance(response, dict):
            count = response.get("count")

        if count and count > 0:
            logging.info(f"Content matching filter {filter} found recently. Skipping.")
            return True
        return False
    except Exception as e:
        logging.error(f"Error checking Supabase for recent embedding: {e}")
        # Default to re-processing on error
        return False


async def url_exists_in_db(url: str) -> bool:
    """Checks if any document with the given source URL already exists.

    This is a cross-run de-duplication helper. If a resource with the exact
    same `metadata.source_url` is present in the `documents` table, return True.
    """
    if not supabase:
        # If Supabase isn't configured, avoid blocking the pipeline
        return False
    
    if not url:
        return False

    try:
        def _execute_query():
            return (
                supabase
                .table("documents")
                .select("id", count="exact")
                .eq("metadata->>source_url", url)
                .limit(1)
                .execute()
            )

        response = await asyncio.to_thread(_execute_query)

        count = getattr(response, "count", None)
        if count is None and isinstance(response, dict):
            count = response.get("count")

        return bool(count and count > 0)
    except Exception as e:
        logging.error(f"Error checking Supabase for existing URL '{url}': {e}")
        # Fail-open: if the check fails, allow processing rather than silently skipping
        return False


async def content_has_changed(url: str, new_hash: str) -> bool:
    """Return True when the stored content hash differs from the new hash."""
    if not url or not new_hash:
        return True
    if not supabase:
        return True

    try:
        def _execute_query():
            return (
                supabase
                .table("documents")
                .select("metadata")
                .eq("metadata->>source_url", url)
                .limit(1)
                .execute()
            )

        response = await asyncio.to_thread(_execute_query)
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")

        if not data:
            return True

        first_record = data[0] if isinstance(data, list) else data
        stored_metadata = (first_record or {}).get("metadata") or {}
        stored_hash = stored_metadata.get("content_hash")
        if stored_hash is None:
            return True
        return stored_hash != new_hash
    except Exception as e:
        logging.error(f"Error comparing content hash for '{url}': {e}")
        return True


# --- Synchronous helpers for harvesting pipeline ---
def check_if_embedded_recently_sync(filter: Dict[str, Any], days: int = 2) -> bool:
    """Synchronous variant to check for recently embedded content.

    Mirrors check_if_embedded_recently but executes the Supabase query directly
    without asyncio to allow use from synchronous Selenium pipeline.
    """
    if not supabase:
        return False
    if not isinstance(filter, dict):
        return False

    try:
        cutoff_dt = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_iso = cutoff_dt.isoformat()

        query = supabase.table("documents").select("metadata", count="exact")
        if "source_url" in filter and filter["source_url"]:
            query = query.eq("metadata->>source_url", filter["source_url"])
        elif filter:
            query = query.contains("metadata", filter)
        query = query.gt("metadata->>retrieval_date", cutoff_iso).limit(1)
        response = query.execute()
        count = getattr(response, "count", None)
        if count is None and isinstance(response, dict):
            count = response.get("count")
        return bool(count and count > 0)
    except Exception as e:
        logging.error(f"Error (sync) checking Supabase for recent embedding: {e}")
        return False


def url_exists_in_db_sync(url: str) -> bool:
    """Synchronous check for existing document by source_url."""
    if not supabase:
        return False
    if not url:
        return False
    try:
        response = (
            supabase
            .table("documents")
            .select("id", count="exact")
            .eq("metadata->>source_url", url)
            .limit(1)
            .execute()
        )
        count = getattr(response, "count", None)
        if count is None and isinstance(response, dict):
            count = response.get("count")
        return bool(count and count > 0)
    except Exception as e:
        logging.error(f"Error (sync) checking Supabase for existing URL '{url}': {e}")
        return False


def content_has_changed_sync(url: str, new_hash: str) -> bool:
    """Synchronous variant of content_has_changed."""
    if not url or not new_hash:
        return True
    if not supabase:
        return True

    try:
        response = (
            supabase
            .table("documents")
            .select("metadata")
            .eq("metadata->>source_url", url)
            .limit(1)
            .execute()
        )
        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")

        if not data:
            return True

        first_record = data[0] if isinstance(data, list) else data
        stored_metadata = (first_record or {}).get("metadata") or {}
        stored_hash = stored_metadata.get("content_hash")
        if stored_hash is None:
            return True
        return stored_hash != new_hash
    except Exception as e:
        logging.error(f"Error (sync) comparing content hash for '{url}': {e}")
        return True


def delete_documents_by_source_url(url: str) -> None:
    """Delete all documents whose metadata.source_url matches the provided URL."""
    if not supabase or not url:
        return

    try:
        (
            supabase
            .table("documents")
            .delete()
            .eq("metadata->>source_url", url)
            .execute()
        )
        logging.info("Deleted existing documents for source URL %s", url)
    except Exception as e:
        logging.error(f"Failed to delete documents for '{url}': {e}")
        raise


# --- Optional: Test Initialization ---
if __name__ == "__main__":
    logging.info("\n--- Running local initialization test for embedding.py ---")
    if supabase and embeddings_model and vector_store:
        logging.info("✅ Initialization appears successful.")
    else:
        logging.error("❌ Initialization failed. Check error messages above.")