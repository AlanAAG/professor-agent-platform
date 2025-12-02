# src/refinery/embedding.py

import os
import logging
import datetime
import asyncio
from difflib import SequenceMatcher  # <--- Critical for title matching
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from typing import List, Dict, Any

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
# Default to Gemini 768-dim model
EMBEDDING_MODEL_NAME = "text-embedding-004"
EXPECTED_DIM = 768

# --- 1. Initialize Clients ---
supabase = None
embeddings_model = None
vector_store = None 
llm_summarizer = None

try:
    # --- Check and Load Environment Variables ---
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    # Support both key names to prevent "API_KEY_INVALID" errors
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    if not gemini_api_key:
        # Warning only; allows importing this module for typing/testing without crashing immediately
        logging.warning("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment.")
    else:
        # --- Initialize Supabase Client ---
        supabase = create_client(supabase_url, supabase_key)
        logging.info("Supabase client initialized.")

        # --- Initialize Gemini Embedding Model ---
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=gemini_api_key,
        )
        logging.info(f"Gemini embeddings initialized: {EMBEDDING_MODEL_NAME}")

        # --- Initialize LLM Summarizer ---
        try:
            llm_summarizer = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=gemini_api_key,
                temperature=0.3
            )
            logging.info("Gemini summarizer initialized: gemini-2.5-flash")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini summarizer: {e}")
            llm_summarizer = None

        # Validate embedding dimensions match database schema
        try:
            # Perform a quick test query to verify dimensions
            test_embedding = embeddings_model.embed_query("dimension validation test")
            if len(test_embedding) != EXPECTED_DIM:
                raise ValueError(
                    f"Embedding model {EMBEDDING_MODEL_NAME} produces {len(test_embedding)}-dimensional "
                    f"vectors, but database expects {EXPECTED_DIM} (Gemini Standard). Please reset your DB."
                )
            logging.info(f"✓ Embedding dimension validated: {EXPECTED_DIM}")
        except Exception as e:
            logging.error(f"Embedding dimension validation failed: {e}")
            raise

        # --- Initialize the Vector Store Client ---
        vector_store = SupabaseVectorStore(
            client=supabase,
            table_name="documents_v2",
            embedding=embeddings_model
        )
        logging.info("SupabaseVectorStore initialized.")

except Exception as e:
    logging.error("Could not initialize embedding clients.")
    logging.error(f"Error details: {e}")

# --- 3. Initialize the Text Splitter ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=200, 
    length_function=len
)
logging.info("Text splitter initialized.")


# --- Metadata Validation ---
REQUIRED_METADATA = ["class_name", "content_type"]

def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and normalizes metadata before embedding."""
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary.")

    for field in REQUIRED_METADATA:
        if field not in metadata:
            raise ValueError(f"Missing required metadata field: {field}")

    if "retrieval_date" not in metadata:
        metadata["retrieval_date"] = datetime.datetime.now().isoformat()

    if "content_hash" in metadata and metadata["content_hash"] is not None:
        metadata["content_hash"] = str(metadata["content_hash"])

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


def _generate_context_summary(full_text: str) -> str:
    """Generates a context summary using LLM."""
    if not llm_summarizer:
        return ""

    try:
        # Truncate to preserve tokens
        truncated_text = full_text[:30000]

        prompt = ChatPromptTemplate.from_template(
            """You are an expert archivist. Analyze the following document and provide a concise context summary (max 50 words).
Focus on: What is the main topic? Who is the intended audience? What are the key entities (concepts, formulas)?
Do NOT summarize the content details, just the context.

Document Start:
{text}

Context Summary:"""
        )

        chain = prompt | llm_summarizer | StrOutputParser()
        return chain.invoke({"text": truncated_text})

    except Exception as e:
        logging.error(f"Error generating context summary: {e}")
        return ""


# --- Main Functions ---
def chunk_and_embed_text(clean_text: str, metadata: Dict[str, Any]):
    if not vector_store:
        raise EnvironmentError("Vector store is not initialized.")
    if not clean_text:
        return

    logging.info(f"-> Chunking and embedding for: {metadata.get('class_name')}")

    global_context = _generate_context_summary(clean_text)

    # --- Contextual Header Construction ---
    header_parts = []

    if global_context:
        header_parts.append(f"Context Summary: {global_context}")

    # Course
    course = metadata.get("class_name") or "Unknown Context"
    header_parts.append(f"Course: {course}")

    # Source
    source = metadata.get("title")
    if source:
        header_parts.append(f"Source: {source}")
    else:
        header_parts.append("Source: General")

    # Instructor
    instructor = metadata.get("teacher_name")
    if instructor:
        header_parts.append(f"Instructor: {instructor}")

    # Date
    date_val = metadata.get("lecture_date")
    if date_val:
        header_parts.append(f"Date: {date_val}")

    header_parts.append("---")
    context_header = "\n".join(header_parts) + "\n"

    documents = text_splitter.create_documents([clean_text])
    
    normalized_metadata = validate_metadata(metadata)
    for doc in documents:
        doc.page_content = context_header + doc.page_content
        doc.metadata = normalized_metadata

    try:
        _add_documents_with_retry(documents)
        logging.info(f"✅ Embedding successful for {len(documents)} chunks.")
    except Exception as e:
        logging.error(f"❌ Failed after retries: {e}")
        raise

async def check_if_embedded_recently(filter: Dict[str, Any], days: int = 7) -> bool:
    if not supabase: return False
    try:
        cutoff_dt = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_iso = cutoff_dt.isoformat()
        query = supabase.table("documents_v2").select("metadata", count="exact")
        if "source_url" in filter and filter["source_url"]:
            query = query.eq("metadata->>source_url", filter["source_url"])
        elif filter:
            query = query.contains("metadata", filter)
        query = query.gt("metadata->>retrieval_date", cutoff_iso).limit(1)
        response = await asyncio.to_thread(query.execute)
        count = getattr(response, "count", None) or (response.get("count") if isinstance(response, dict) else 0)
        return bool(count and count > 0)
    except Exception as e:
        logging.error(f"Error checking Supabase: {e}")
        return False

# Sync Helpers
def check_if_embedded_recently_sync(filter: Dict[str, Any], days: int = 2) -> bool:
    return asyncio.run(check_if_embedded_recently(filter, days))

def url_exists_in_db_sync(url: str) -> bool:
    if not supabase or not url: return False
    try:
        response = supabase.table("documents_v2").select("id", count="exact").eq("metadata->>source_url", url).limit(1).execute()
        count = getattr(response, "count", None) or (response.get("count") if isinstance(response, dict) else 0)
        return bool(count and count > 0)
    except Exception as e:
        logging.error(f"Error checking URL {url}: {e}")
        return False

def content_has_changed_sync(url: str, new_hash: str) -> bool:
    if not supabase or not url: return True
    try:
        response = supabase.table("documents_v2").select("metadata").eq("metadata->>source_url", url).limit(1).execute()
        data = getattr(response, "data", []) or (response.get("data") if isinstance(response, dict) else [])
        if not data: return True
        return data[0].get("metadata", {}).get("content_hash") != new_hash
    except Exception:
        return True

def delete_documents_by_source_url(url: str) -> None:
    if not supabase or not url: return
    try:
        supabase.table("documents_v2").delete().eq("metadata->>source_url", url).execute()
        logging.info(f"Deleted existing documents for {url}")
    except Exception as e:
        logging.error(f"Failed to delete {url}: {e}")

# --- RESTORED: Manual Transcript Helper (Critical for Harvester) ---
def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

async def find_manual_transcript_by_metadata(title: str, date_iso: str) -> bool:
    """Checks if a manual transcript exists to skip Whisper transcription."""
    if not supabase or not date_iso:
        return False
    try:
        # 1. Find transcripts on date
        def _execute_query():
            return (
                supabase.table("documents_v2")
                .select("metadata")
                .eq("metadata->>content_type", "manual_transcript")
                .eq("metadata->>lecture_date", date_iso)
                .execute()
            )
        response = await asyncio.to_thread(_execute_query)
        data = getattr(response, "data", []) or (response.get("data") if isinstance(response, dict) else [])
        
        if not data: return False

        # 2. Fuzzy match title
        target_title = (title or "").lower().strip()
        for record in data:
            meta = record.get("metadata", {})
            db_title = (meta.get("title") or "").lower().strip()
            if target_title in db_title or db_title in target_title:
                logging.info(f"Found metadata match: '{target_title}' ~= '{db_title}'")
                return True
            if _similarity(target_title, db_title) > 0.8:
                logging.info(f"Found metadata match (fuzzy): '{target_title}' ~= '{db_title}'")
                return True
        return False
    except Exception as e:
        logging.error(f"Error checking manual transcript: {e}")
        return False

def find_manual_transcript_by_metadata_sync(title: str, date_iso: str) -> bool:
    return asyncio.run(find_manual_transcript_by_metadata(title, date_iso))