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
# Prefer the current public Google GenAI embedding model by default.
# Can be overridden via EMBEDDING_MODEL_NAME env var.
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-004")
# OpenAI embedding model fallback
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


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

    # OPTIMIZATION: Batch embed all query variants in a single API call
    try:
        query_embeddings = embed_queries_batch(queries)
    except Exception as e:
        logging.warning(f"Batch embedding failed, falling back to individual calls: {e}")
        query_embeddings = [None] * len(queries)  # Will trigger individual embedding in retrieve_rag_documents

    # Retrieve candidates for each query using pre-computed embeddings
    results_per_query: List[List[Dict[str, Any]]] = []
    for q, q_embedding in zip(queries, query_embeddings):
        try:
            # Pass pre-computed embedding to avoid redundant API calls
            retrieved = retrieve_rag_documents(
                q, 
                selected_class=class_name if class_name else None, 
                match_count=30, 
                match_threshold=0.7,
                query_embedding=q_embedding
            )
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
    from google import genai
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
_GENAI_CLIENT = None
_OPENAI_CLIENT = None


def _get_supabase_client():
    """Return a cached Supabase client, preferring EXTERNAL_ env vars."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT
    if not create_client:
        raise RuntimeError("Supabase client not available. Install 'supabase'.")

    # Normalize URL to avoid double slashes in requests (e.g., ...co//rest/v1)
    url_raw = os.environ.get("EXTERNAL_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
    url = url_raw.rstrip("/") if url_raw else None
    key = os.environ.get("EXTERNAL_SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Supabase credentials missing. Set EXTERNAL_SUPABASE_URL/EXTERNAL_SUPABASE_SERVICE_KEY or SUPABASE_URL/SUPABASE_KEY",
        )
    _SUPABASE_CLIENT = create_client(url, key)
    return _SUPABASE_CLIENT


def _ensure_genai():
    """Best-effort initialize Google GenAI; don't raise to allow fallbacks."""
    global _GENAI_CLIENT
    try:
        if genai is None:
            return
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return
        if getattr(genai, "Client", None) is not None:
            if _GENAI_CLIENT is None:
                _GENAI_CLIENT = genai.Client(api_key=api_key)
            return
        try:
            if hasattr(genai, "configure"):
                genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        # Swallow to allow OpenAI fallback
        _GENAI_CLIENT = None


def _ensure_openai():
    """Best-effort initialize OpenAI client for embeddings fallback."""
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return
    try:
        # OpenAI Python SDK v1+/v2
        try:
            from openai import OpenAI  # type: ignore
        except Exception:  # pragma: no cover
            return
        # The SDK reads OPENAI_API_KEY from env; no explicit key needed if set
        _OPENAI_CLIENT = OpenAI()
    except Exception:
        _OPENAI_CLIENT = None


def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed a single query string with robust fallbacks (Gemini -> OpenAI)."""
    _ensure_genai()
    # Try the provided model first, then sensible fallbacks
    primary_model = model or EMBEDDING_MODEL_NAME
    candidate_models: list[str] = []
    for m in (primary_model, "text-embedding-004", "models/embedding-001"):
        if m and m not in candidate_models:
            candidate_models.append(m)

    def _extract_single(result: Any) -> list[float]:
        # Common forms
        if isinstance(result, dict):
            if "embedding" in result and isinstance(result["embedding"], list):
                # Either a single vector or batch
                emb = result["embedding"]
                if emb and isinstance(emb[0], (int, float)):
                    return emb  # single vector
                if emb and isinstance(emb[0], list):
                    return emb[0]  # first of batch
            if "embeddings" in result and isinstance(result["embeddings"], list):
                first = result["embeddings"][0] if result["embeddings"] else None
                if isinstance(first, dict) and "values" in first:
                    return first["values"]
                if isinstance(first, list):
                    return first
        # Attr-like
        if hasattr(result, "embedding"):
            emb = getattr(result, "embedding")
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                return emb
            if isinstance(emb, list) and emb and isinstance(emb[0], list):
                return emb[0]
        if hasattr(result, "embeddings"):
            embeddings = getattr(result, "embeddings")
            if isinstance(embeddings, list) and embeddings:
                first = embeddings[0]
                if isinstance(first, dict) and "values" in first:
                    return first["values"]
                if isinstance(first, list):
                    return first
        # Index-like
        try:
            emb = result["embedding"]  # type: ignore[index]
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                return emb
            if isinstance(emb, list) and emb and isinstance(emb[0], list):
                return emb[0]
        except Exception:
            pass
        raise RuntimeError(f"Unexpected embed_content result shape: {type(result)}")

    # Try each candidate model across known SDK call styles
    for embedding_model in candidate_models:
        # Try modern client API first (prefer 'contents' per new SDK)
        if _GENAI_CLIENT is not None and getattr(_GENAI_CLIENT, "models", None) is not None:
            try:
                result = _GENAI_CLIENT.models.embed_content(
                    model=embedding_model,
                    contents=[text],  # new SDK prefers 'contents'
                    task_type="retrieval_query",
                )
                return _extract_single(result)
            except TypeError:
                # Older signature that accepts 'content='
                try:
                    result = _GENAI_CLIENT.models.embed_content(
                        model=embedding_model,
                        content=text,
                        task_type="retrieval_query",
                    )
                    return _extract_single(result)
                except Exception:
                    pass
            except Exception:
                # Try alternative API below
                pass

            # Alternative new API: generate_content_embeddings
            try:
                gce = getattr(_GENAI_CLIENT.models, "generate_content_embeddings", None)
                if gce is not None:
                    result = gce(
                        model=embedding_model,
                        requests=[{"content": {"text": text}}],
                    )
                    return _extract_single(result)
            except Exception:
                pass

        # Fallback: module-level API (older libraries or tests)
        try:
            if hasattr(genai, "embed_content"):
                result = genai.embed_content(
                    model=embedding_model,
                    content=text,
                    task_type="retrieval_query",
                )
                return _extract_single(result)
        except Exception:
            pass
    # OpenAI fallback
    _ensure_openai()
    if _OPENAI_CLIENT is not None:
        try:
            resp = _OPENAI_CLIENT.embeddings.create(
                model=os.environ.get("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL),
                input=text,
            )
            data = getattr(resp, "data", None) or []
            if data and hasattr(data[0], "embedding"):
                return data[0].embedding  # type: ignore[return-value]
        except Exception as e:
            logging.warning(f"OpenAI embedding fallback failed: {e}")

    raise RuntimeError("Failed to obtain embedding via available Google GenAI or OpenAI interfaces.")


def embed_queries_batch(texts: List[str], model: str | None = None) -> List[List[float]]:
    """Embed multiple query strings with robust fallbacks (Gemini -> OpenAI)."""
    if not texts:
        return []

    _ensure_genai()
    # Try the provided model first, then sensible fallbacks
    primary_model = model or EMBEDDING_MODEL_NAME
    candidate_models: list[str] = []
    for m in (primary_model, "text-embedding-004", "models/embedding-001"):
        if m and m not in candidate_models:
            candidate_models.append(m)

    def _extract_batch(result: Any) -> List[List[float]]:
        # Dict-like
        if isinstance(result, dict):
            if "embedding" in result and isinstance(result["embedding"], list):
                emb = result["embedding"]
                if emb and isinstance(emb[0], list):
                    return emb  # list of vectors
                if emb and isinstance(emb[0], (int, float)):
                    return [emb]
            if "embeddings" in result and isinstance(result["embeddings"], list):
                out: List[List[float]] = []
                for item in result["embeddings"]:
                    if isinstance(item, dict) and "values" in item:
                        out.append(item["values"])  # type: ignore[list-item]
                    elif isinstance(item, list):
                        out.append(item)
                if out:
                    return out
        # Attr-like
        if hasattr(result, "embedding"):
            emb = getattr(result, "embedding")
            if isinstance(emb, list) and emb:
                if isinstance(emb[0], list):
                    return emb
                if isinstance(emb[0], (int, float)):
                    return [emb]
        if hasattr(result, "embeddings"):
            embeddings = getattr(result, "embeddings")
            if isinstance(embeddings, list) and embeddings:
                out: List[List[float]] = []
                for item in embeddings:
                    if isinstance(item, dict) and "values" in item:
                        out.append(item["values"])  # type: ignore[list-item]
                    elif isinstance(item, list):
                        out.append(item)
                if out:
                    return out
        raise RuntimeError(f"Unexpected batch embed_content result shape: {type(result)}")

    # Try each candidate model across known SDK call styles
    for embedding_model in candidate_models:
        # Try modern client API first (prefer 'contents' per new SDK)
        if _GENAI_CLIENT is not None and getattr(_GENAI_CLIENT, "models", None) is not None:
            try:
                result = _GENAI_CLIENT.models.embed_content(
                    model=embedding_model,
                    contents=texts,
                    task_type="retrieval_query",
                )
                return _extract_batch(result)
            except TypeError:
                try:
                    result = _GENAI_CLIENT.models.embed_content(
                        model=embedding_model,
                        content=texts,
                        task_type="retrieval_query",
                    )
                    return _extract_batch(result)
                except Exception:
                    pass
            except Exception:
                pass

            # Alternative new API: generate_content_embeddings
            try:
                gce = getattr(_GENAI_CLIENT.models, "generate_content_embeddings", None)
                if gce is not None:
                    result = gce(
                        model=embedding_model,
                        requests=[{"content": {"text": t}} for t in texts],
                    )
                    return _extract_batch(result)
            except Exception:
                pass

        # Fallback: module-level API (older libraries or tests)
        try:
            if hasattr(genai, "embed_content"):
                result = genai.embed_content(
                    model=embedding_model,
                    content=texts,
                    task_type="retrieval_query",
                )
                return _extract_batch(result)
        except Exception:
            pass
    # OpenAI batch fallback
    _ensure_openai()
    if _OPENAI_CLIENT is not None:
        try:
            resp = _OPENAI_CLIENT.embeddings.create(
                model=os.environ.get("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL),
                input=texts,
            )
            data = getattr(resp, "data", None) or []
            if data:
                # Ensure order aligns with inputs
                vectors: List[List[float]] = []
                # Some SDK versions return a list where each item has an 'index'
                # We'll sort to be safe, then map to embeddings
                try:
                    data_sorted = sorted(data, key=lambda d: getattr(d, "index", 0))
                except Exception:
                    data_sorted = data
                for item in data_sorted:
                    vec = getattr(item, "embedding", None)
                    if isinstance(vec, list):
                        vectors.append(vec)
                if vectors:
                    return vectors
        except Exception as e:
            logging.warning(f"OpenAI batch embedding fallback failed: {e}")

    # Ultimate fallback: individual calls
    logging.warning("Batch embedding failed, falling back to individual calls")
    return [embed_query(text, model) for text in texts]


def retrieve_rag_documents(
    query: str,
    selected_class: str | None = None,
    match_count: int = 20,
    match_threshold: float = 0.7,
    query_embedding: List[float] | None = None,
) -> list[dict]:
    """Retrieve documents via Supabase 'match_documents' RPC.
    
    Args:
        query: Query string (used for embedding if query_embedding not provided)
        selected_class: Optional class filter
        match_count: Number of documents to retrieve
        match_threshold: Similarity threshold
        query_embedding: Pre-computed embedding (if provided, skips embedding step)
    """
    supabase = _get_supabase_client()
    
    # Use pre-computed embedding if provided, otherwise compute it
    if query_embedding is None:
        try:
            query_embedding = embed_query(query)
        except Exception as e:
            logging.warning(f"Query embedding failed; skipping retrieval: {e}")
            return []
    
    payload = {
        "query_embedding": query_embedding,
        "match_threshold": float(match_threshold),
        "match_count": int(match_count),
    }
    if selected_class:
        payload["filter_class"] = selected_class
    response = supabase.rpc("match_documents", payload).execute()
    return getattr(response, "data", None) or []


def retrieve_rag_documents_keyword_fallback(
    query: str,
    selected_class: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Lightweight keyword fallback against Supabase when embeddings/RPC return nothing.

    Performs a case-insensitive search over `documents.content` and optionally filters
    by `metadata->>class_name`. This is less precise than vector search but ensures
    we can still surface relevant material directly from the database when embeddings
    are unavailable or empty.
    """
    try:
        supabase = _get_supabase_client()
    except Exception:
        return []

    # Build a base query once
    def _base_query():
        q0 = supabase.table("documents").select("id, content, metadata, created_at")
        if selected_class:
            q0 = q0.filter("metadata->>class_name", "eq", selected_class)
        return q0

    # Sanitize and bound the query to reduce chances of 400s due to weird tokens
    qtext = (query or "").strip()
    # Cap length to avoid overly long LIKE/FTS payloads
    if len(qtext) > 200:
        qtext = qtext[:200]

    # Try multiple strategies to avoid PostgREST 400s across environments.
    # Prefer full-text search variants first, then fall back to ILIKE.
    strategies = [
        # Web-search (handles natural language best)
        ("text_search_web", lambda q: q.text_search("content", qtext, config="english", type="websearch")),
        # Phrase full-text search
        ("filter_phfts",     lambda q: q.filter("content", "phfts", qtext)),
        # Plain full-text search
        ("filter_fts",       lambda q: q.filter("content", "fts", qtext)),
        # Web full-text search operator
        ("filter_wfts",      lambda q: q.filter("content", "wfts", qtext)),
        # Percent-based ILIKE (widely supported)
        ("ilike_percent",    lambda q: q.ilike("content", f"%{qtext}%")),
        ("filter_ilike_pct", lambda q: q.filter("content", "ilike", f"%{qtext}%")),
        # Star-based ILIKE (some PostgREST deployments accept this form)
        ("filter_ilike_star", lambda q: q.filter("content", "ilike", f"*{qtext}*")),
    ]

    for _name, apply_strategy in strategies:
        try:
            q = apply_strategy(_base_query())
            q = q.order("created_at", desc=True).limit(max(1, int(limit)))
            resp = q.execute()
            data = getattr(resp, "data", None) or []
            if not data:
                # Try next strategy if empty
                continue

            # Normalize to match RPC shape as much as possible
            norm: list[dict] = []
            for d in data:
                meta = d.get("metadata") or {}
                norm.append(
                    {
                        "id": d.get("id"),
                        "content": d.get("content", ""),
                        "metadata": meta,
                        "class_name": meta.get("class_name"),
                        "title": meta.get("title"),
                        "section": meta.get("section"),
                        "url": meta.get("url") or meta.get("source_url"),
                        "similarity": 0.0,
                    }
                )
            return norm
        except Exception:
            # Try the next strategy on any error (including HTTP 400)
            continue

    # If all strategies fail or return empty, return []
    return []


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
def parse_general_date(date_str: str, local_tz: tz.tzfile = tz.gettz('UTC')) -> Optional[datetime.datetime]:
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

        # If the parsed datetime is naive, assume UTC to avoid date shifts
        if dt_obj.tzinfo is None:
            assumed_tz = local_tz or tz.gettz('UTC')
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