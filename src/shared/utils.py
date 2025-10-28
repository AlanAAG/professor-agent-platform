# src/shared/utils.py

import re
import os
import logging
import math
import numpy as np
from typing import List, Dict, Any, Optional
import datetime
from dateutil import parser # Use dateutil for flexible parsing
from dateutil import tz

# Shared constants
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "models/embedding-001")

# --- Helper Functions for RRF/MMR/Boosting ---

def expand_query(query: str) -> List[str]:
    """Returns the original query plus at least 3 fixed, rule-based variations to improve initial retrieval diversity."""
    expanded_queries = [query]  # Original query
    
    # Add rule-based variations
    expanded_queries.append(f"explain {query}")
    expanded_queries.append(f"definition of {query}")
    expanded_queries.append(f"examples of {query}")
    expanded_queries.append(f"how to {query}")
    
    return expanded_queries


def apply_rrf_and_boost(results_per_query: List[List[Dict]], query: str, class_name: str) -> Dict[str, float]:
    """
    Calculates the RRF score for each document, merging results from the original query and its expanded variations.
    Applies metadata boosting rules to the fused scores.
    Returns a dictionary mapping document ID to final boosted RRF score.
    """
    doc_scores = {}
    
    # Calculate RRF scores: score(doc) = sum(1 / (60 + rank_in_query_i))
    for query_results in results_per_query:
        for rank, doc in enumerate(query_results):
            doc_id = doc.get('id', str(hash(doc.get('content', ''))))
            rrf_score = 1.0 / (60 + rank)
            
            if doc_id in doc_scores:
                doc_scores[doc_id] += rrf_score
            else:
                doc_scores[doc_id] = rrf_score
    
    # Apply metadata boosting rules
    for query_results in results_per_query:
        for doc in query_results:
            doc_id = doc.get('id', str(hash(doc.get('content', ''))))
            if doc_id not in doc_scores:
                continue
                
            # Exact class match: +0.15
            if class_name and doc.get('class_name') == class_name:
                doc_scores[doc_id] += 0.15
            
            # Section priority boosting
            section = doc.get('section', '').lower()
            if 'sessions' in section:
                doc_scores[doc_id] += 0.10  # Section priority (sessions): +0.10
            elif 'in_class' in section:
                doc_scores[doc_id] += 0.05  # Section priority (in_class): +0.05
    
    return doc_scores


def compute_cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """Compute cosine similarity between two embeddings using numpy."""
    emb1_np = np.array(emb1)
    emb2_np = np.array(emb2)
    
    dot_product = np.dot(emb1_np, emb2_np)
    norm1 = np.linalg.norm(emb1_np)
    norm2 = np.linalg.norm(emb2_np)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def maximal_marginal_relevance(
    query_embedding: List[float], 
    all_documents: Dict[str, Dict], 
    doc_scores: Dict[str, float], 
    lambda_param: float = 0.7, 
    k: int = 7
) -> List[Dict]:
    """
    Uses MMR to select top k documents that are relevant to the query AND different from already-selected set.
    Formula: score = λ * relevance - (1 - λ) * max_similarity_to_selected
    """
    if not doc_scores or not all_documents:
        return []
    
    selected_docs = []
    remaining_doc_ids = set(doc_scores.keys())
    
    for _ in range(min(k, len(remaining_doc_ids))):
        best_doc_id = None
        best_mmr_score = float('-inf')
        
        for doc_id in remaining_doc_ids:
            if doc_id not in all_documents:
                continue
                
            doc = all_documents[doc_id]
            relevance_score = doc_scores[doc_id]
            
            # Calculate max similarity to already selected documents
            max_similarity = 0.0
            if selected_docs:
                doc_embedding = doc.get('embedding', [])
                if doc_embedding:
                    for selected_doc in selected_docs:
                        selected_embedding = selected_doc.get('embedding', [])
                        if selected_embedding:
                            similarity = compute_cosine_similarity(doc_embedding, selected_embedding)
                            max_similarity = max(max_similarity, similarity)
            
            # MMR formula: λ * relevance - (1 - λ) * max_similarity_to_selected
            mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_doc_id = doc_id
        
        if best_doc_id:
            selected_docs.append(all_documents[best_doc_id])
            remaining_doc_ids.remove(best_doc_id)
    
    return selected_docs


def cohere_rerank(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Orchestrator function that replaces the old cohere_rerank.
    Implements RRF/MMR/Boosting pipeline:
    1. Expand queries
    2. Embed queries  
    3. Retrieve documents for each expanded query
    4. Apply RRF and boosting
    5. Apply MMR for final selection
    """
    if not documents:
        return documents
    
    try:
        # Extract class_name from the first document if available
        class_name = ""
        if documents and isinstance(documents[0], dict):
            class_name = documents[0].get('class_name', '')
        
        # Step 1: Expand queries
        expanded_queries = expand_query(query)
        
        # Step 2: Embed queries
        query_embeddings = {}
        for expanded_query in expanded_queries:
            try:
                query_embeddings[expanded_query] = embed_query(expanded_query)
            except Exception as e:
                logging.warning(f"Failed to embed query '{expanded_query}': {e}")
                continue
        
        # Step 3: Retrieve documents for each expanded query
        results_per_query = []
        all_documents = {}  # doc_id -> document dict
        
        for expanded_query in expanded_queries:
            if expanded_query not in query_embeddings:
                continue
                
            try:
                # Retrieve fresh candidates (top 30 per query for RRF)
                retrieved_docs = retrieve_rag_documents(
                    query=expanded_query,
                    selected_class=class_name if class_name else None,
                    match_count=30,
                    match_threshold=0.7
                )
                
                results_per_query.append(retrieved_docs)
                
                # Build all_documents mapping
                for doc in retrieved_docs:
                    doc_id = doc.get('id', str(hash(doc.get('content', ''))))
                    all_documents[doc_id] = doc
                    
            except Exception as e:
                logging.warning(f"Failed to retrieve documents for query '{expanded_query}': {e}")
                continue
        
        # If no results were retrieved, fall back to original documents
        if not results_per_query:
            logging.warning("No documents retrieved from expanded queries, using original documents")
            return documents[:7]  # Return top 7 from original
        
        # Step 4: Apply RRF and boosting
        doc_scores = apply_rrf_and_boost(results_per_query, query, class_name)
        
        if not doc_scores:
            logging.warning("No document scores calculated, using original documents")
            return documents[:7]
        
        # Step 5: Apply MMR for final selection
        original_query_embedding = query_embeddings.get(query, [])
        if not original_query_embedding:
            # If we can't get the original query embedding, just return top scored docs
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            result_docs = []
            for doc_id, _ in sorted_docs[:7]:
                if doc_id in all_documents:
                    result_docs.append(all_documents[doc_id])
            return result_docs
        
        final_docs = maximal_marginal_relevance(
            query_embedding=original_query_embedding,
            all_documents=all_documents,
            doc_scores=doc_scores,
            lambda_param=0.7,
            k=7
        )
        
        return final_docs
        
    except Exception as e:
        logging.error(f"Error in cohere_rerank orchestrator: {e}")
        # Fallback to original documents
        return documents[:7] if len(documents) > 7 else documents

 
# --- Shared RAG Retrieval (Supabase RPC + Gemini embeddings) ---
try:
    import google.generativeai as genai
except Exception:
    genai = None  # Optional dependency

try:
    from supabase import create_client, Client
except Exception:
    create_client = None  # type: ignore
    Client = None  # type: ignore

try:
    from langchain_core.documents import Document
except Exception:
    Document = None  # type: ignore

_SUPABASE_CLIENT = None


def _get_supabase_client():
    """Return a cached Supabase client, preferring EXTERNAL_ env vars."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT
    if not create_client:
        raise RuntimeError("Supabase client not available. Install 'supabase'.")

    url = os.environ.get("EXTERNAL_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    key = os.environ.get("EXTERNAL_SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Supabase credentials missing. Set EXTERNAL_SUPABASE_URL/EXTERNAL_SUPABASE_SERVICE_KEY or SUPABASE_URL/SUPABASE_KEY",
        )
    _SUPABASE_CLIENT = create_client(url, key)
    return _SUPABASE_CLIENT


def _ensure_genai():
    if genai is None:
        raise RuntimeError("google-generativeai not available. Install 'google-generativeai'.")
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY for embeddings.")
    genai.configure(api_key=api_key)


def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed a query string using Gemini embeddings."""
    _ensure_genai()
    embedding_model = model or EMBEDDING_MODEL_NAME
    result = genai.embed_content(
        model=embedding_model,
        content=text,
        task_type="retrieval_query",
    )
    return result.get("embedding") or result["embedding"]


def retrieve_rag_documents(
    query: str,
    selected_class: str | None = None,
    match_count: int = 20,
    match_threshold: float = 0.7,
) -> list[dict]:
    """Retrieve documents via Supabase 'match_documents' RPC."""
    supabase = _get_supabase_client()
    query_embedding = embed_query(query)
    payload = {
        "query_embedding": query_embedding,
        "match_threshold": float(match_threshold),
        "match_count": int(match_count),
    }
    if selected_class:
        payload["filter_class"] = selected_class
    response = supabase.rpc("match_documents", payload).execute()
    return getattr(response, "data", None) or []


def _to_langchain_documents(raw_docs: list[dict]) -> list[Document]:
    if Document is None:
        raise RuntimeError("langchain-core is required to build Document objects.")
    docs: list[Document] = []
    for d in raw_docs:
        content = d.get("page_content") or d.get("content") or ""
        meta = d.get("metadata") or {}
        for key in ("class_name", "title", "section", "url", "id", "similarity"):
            if key in d and key not in meta:
                meta[key] = d[key]
        docs.append(Document(page_content=content, metadata=meta))
    return docs


def retrieve_rag_documents_langchain(
    query: str,
    selected_class: str | None = None,
    match_count: int = 20,
    match_threshold: float = 0.7,
):
    """Retrieve documents and return as LangChain Document objects."""
    raw = retrieve_rag_documents(
        query=query,
        selected_class=selected_class,
        match_count=match_count,
        match_threshold=match_threshold,
    )
    return _to_langchain_documents(raw)


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