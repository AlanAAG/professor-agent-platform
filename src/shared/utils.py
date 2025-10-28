# src/shared/utils.py

import re
import os
import math
import hashlib
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import datetime
from dateutil import parser # Use dateutil for flexible parsing
from dateutil import tz

# Shared constants
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "models/embedding-001")


# --- Helper Functions for RRF/MMR/Boosting (Option A) ---

def expand_query(query: str) -> List[str]:
    """Return original query plus simple rule-based variants to diversify retrieval."""
    base = query.strip()
    if not base:
        return [""]
    variants = [
        base,
        f"{base} explanation",
        f"{base} definition",
        f"{base} examples",
        f"{base} in practice",
    ]
    # Ensure uniqueness while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _get_doc_id(doc: Dict[str, Any]) -> str:
    """Best-effort stable identifier for a document dictionary."""
    for key in ("id", "doc_id", "chunk_id"):
        if key in doc and doc[key] is not None:
            return str(doc[key])
    meta = doc.get("metadata") or {}
    for key in ("id", "doc_id", "chunk_id"):
        if key in meta and meta[key] is not None:
            return str(meta[key])
    # Fallback: hash of core fields
    basis = (doc.get("url") or "") + "|" + (doc.get("title") or "") + "|" + (doc.get("section") or "") + "|" + (doc.get("content") or doc.get("page_content") or "")
    return hashlib.md5(basis.encode("utf-8")).hexdigest()


def _get_doc_class(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    return str(doc.get("class_name") or meta.get("class_name") or "").strip()


def _get_doc_section(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    return str(doc.get("section") or meta.get("section") or "").strip().lower()


def apply_rrf_and_boost(results_per_query: List[List[Dict]], query: str, class_name: str) -> Dict[str, float]:
    """Fuse ranked lists via RRF and apply simple metadata boosts.

    Returns mapping of doc_id -> final score.
    """
    # RRF accumulation
    fused_scores: Dict[str, float] = {}
    for result_list in results_per_query:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = _get_doc_id(doc)
            # RRF formula: score(doc) = sum(1 / (60 + rank_in_query_i))
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (60.0 + float(rank))

    # Metadata boosts
    # Note: Iterate through all documents to find boostable metadata
    for result_list in results_per_query:
        for doc in result_list:
            doc_id = _get_doc_id(doc)
            if doc_id not in fused_scores:
                continue
            boost = 0.0
            # Exact class match: +0.15
            if class_name and _get_doc_class(doc) == class_name:
                boost += 0.15
            # Section priorities (sessions > in_class > others)
            section = _get_doc_section(doc)
            if section == "sessions":
                boost += 0.10
            elif section == "in_class":
                boost += 0.05
            fused_scores[doc_id] += boost

    return fused_scores


def compute_cosine_similarity(embedding_a: List[float], embedding_b: List[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if invalid."""
    try:
        a = np.asarray(embedding_a, dtype=float)
        b = np.asarray(embedding_b, dtype=float)
        if a.ndim != 1 or b.ndim != 1:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        result = float(np.dot(a, b) / denom)
        if not math.isfinite(result):
            return 0.0
        return result
    except Exception:
        return 0.0


def _extract_doc_embedding(doc: Dict[str, Any]) -> List[float] | None:
    """Extract embedding from document dict, if present."""
    # Common keys returned by the Supabase RPC
    for key_path in (
        ("embedding",),
        ("metadata", "embedding"),
        ("embedding_vector",),
        ("metadata", "embedding_vector"),
        ("vector",),
        ("metadata", "vector"),
    ):
        cur: Any = doc
        for key in key_path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                cur = None
                break
        if isinstance(cur, list) and len(cur) > 0:
            return cur  # type: ignore[return-value]
    return None


def maximal_marginal_relevance(
    query_embedding: List[float],
    all_documents: Dict[str, Dict],
    doc_scores: Dict[str, float],
    lambda_param: float = 0.7,
    k: int = 7,
) -> List[Dict]:
    """Select a diverse top-k set using MMR over boosted RRF relevance."""
    if not doc_scores:
        return []

    # Prepare candidate set sorted by relevance (boosted RRF score) first
    candidates = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    selected: List[str] = []
    selected_docs: List[Dict] = []

    # Cache document embeddings for rapid similarity lookups
    embedding_cache: Dict[str, List[float] | None] = {}
    for doc_id, _ in candidates:
        doc = all_documents.get(doc_id)
        if doc is None:
            embedding_cache[doc_id] = None
        else:
            embedding_cache[doc_id] = _extract_doc_embedding(doc)

    while len(selected) < min(k, len(candidates)):
        best_doc_id = None
        best_score = -float("inf")
        for doc_id, rel in candidates:
            if doc_id in selected:
                continue
            # Relevance term from boosted RRF
            relevance = rel
            # Diversity term: max similarity to any already selected
            if not selected:
                diversity_penalty = 0.0
            else:
                cand_emb = embedding_cache.get(doc_id)
                if not cand_emb:
                    # Without an embedding, we cannot compute similarity; assume zero similarity
                    diversity_penalty = 0.0
                else:
                    max_sim = 0.0
                    for sel_id in selected:
                        sel_emb = embedding_cache.get(sel_id)
                        if sel_emb:
                            max_sim = max(max_sim, compute_cosine_similarity(cand_emb, sel_emb))
                    # MMR diversity penalty calculation
                    diversity_penalty = (1.0 - lambda_param) * max_sim
            
            # MMR Score: (Lambda * Relevance) - ( (1 - Lambda) * Max Similarity)
            score = lambda_param * relevance - diversity_penalty
            
            if score > best_score:
                best_score = score
                best_doc_id = doc_id
        
        if best_doc_id is None:
            break
        selected.append(best_doc_id)
        selected_docs.append(all_documents[best_doc_id])

    return selected_docs


def _normalize_doc_input(obj: Any) -> Dict[str, Any]:
    """Normalize various document inputs (dict, LangChain Document, str) to a dict."""
    if isinstance(obj, dict):
        return obj
    # LangChain Document-like
    if hasattr(obj, "page_content") and hasattr(obj, "metadata"):
        return {
            "content": getattr(obj, "page_content", ""),
            "metadata": getattr(obj, "metadata", {}) or {},
        }
    if isinstance(obj, str):
        return {"content": obj, "metadata": {}}
    # Fallback
    return {"content": str(obj), "metadata": {}}


def cohere_rerank(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reranking orchestrator using Query Expansion + RRF + Boosting + MMR.
    
    This function acts as a zero-cost drop-in replacement for Cohere.
    """
    # Track whether caller passed LangChain Documents (for faithful return type)
    input_is_langchain = bool(documents) and hasattr(documents[0], "page_content") and hasattr(documents[0], "metadata")
    # Normalize input docs in case caller passes LangChain Documents
    normalized_input_docs: List[Dict[str, Any]] = [_normalize_doc_input(d) for d in (documents or [])]
    
    class_name = ""
    # Attempt to infer class from provided docs as a fallback boost context
    for d in normalized_input_docs:
        c = _get_doc_class(d)
        if c:
            class_name = c
            break

    # Expand queries
    queries = expand_query(query)

    # Retrieve candidates for each query
    results_per_query: List[List[Dict[str, Any]]] = []
    for q in queries:
        try:
            # We call retrieve_rag_documents with the expanded query
            retrieved = retrieve_rag_documents(q, selected_class=class_name if class_name else None, match_count=30, match_threshold=0.7)
        except Exception as e:
            logging.warning(f"Retrieval failed for variant '{q}': {e}")
            retrieved = []
        results_per_query.append(retrieved)

    # Build master map of id -> doc for all retrieved
    all_docs_map: Dict[str, Dict[str, Any]] = {}
    for lst in results_per_query:
        for d in lst:
            all_docs_map.setdefault(_get_doc_id(d), d)
    # Also include any originals not in retrieval results
    for d in normalized_input_docs:
        all_docs_map.setdefault(_get_doc_id(d), d)

    # Compute RRF with boosting
    fused_scores = apply_rrf_and_boost(results_per_query, query=query, class_name=class_name)

    # --- Fallback: Return top RRF score only if query embedding fails ---
    if not fused_scores:
        # Fall back to original documents if retrieval was completely empty
        selected_dicts = normalized_input_docs[:7]
        if input_is_langchain and Document is not None:
            return _to_langchain_documents(selected_dicts)  # type: ignore[return-value]
        return selected_dicts

    # Embed the original query for MMR
    try:
        q_emb = embed_query(query)
    except Exception as e:
        logging.warning(f"Query embedding failed for MMR; returning top by RRF only: {e}")
        # If embedding fails, return top-7 by fused score without MMR
        top_ids = [doc_id for doc_id, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:7]]
        selected_dicts = [all_docs_map[i] for i in top_ids if i in all_docs_map]
        if input_is_langchain and Document is not None:
            return _to_langchain_documents(selected_dicts)  # type: ignore[return-value]
        return selected_dicts

    # --- Final Step: Run MMR selection ---
    mmr_selected_dicts = maximal_marginal_relevance(q_emb, all_docs_map, fused_scores, lambda_param=0.7, k=7)
    
    # Return in the format the caller expected (dict or LangChain Document)
    if input_is_langchain and Document is not None:
        return _to_langchain_documents(mmr_selected_dicts)  # type: ignore[return-value]
    return mmr_selected_dicts

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