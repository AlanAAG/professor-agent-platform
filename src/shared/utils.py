# src/shared/utils.py

import re
import os
import math
import hashlib
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable
import datetime
from dateutil import parser # Use dateutil for flexible parsing
from dateutil import tz

try:
    from postgrest.exceptions import APIError as PostgrestAPIError
except Exception:  # pragma: no cover - optional dependency in some environments
    PostgrestAPIError = None  # type: ignore[assignment]

try:
    import sentry_sdk
except Exception:  # pragma: no cover - optional dependency
    sentry_sdk = None  # type: ignore[assignment]

# Shared constants
# Prefer the current public Google GenAI embedding model by default.
# Can be overridden via EMBEDDING_MODEL_NAME env var.
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-004")
# CRITICAL: This dimension must match the database vector column definition
# See database/match_documents.sql line 5: vector(768)
# Compatible models: text-embedding-004 (768), models/embedding-001 (768)
EXPECTED_EMBEDDING_DIM = 768
# OpenAI embedding model fallback
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DISABLE_OPENAI_FALLBACK = os.environ.get("DISABLE_OPENAI_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}
SENTRY_DSN = os.environ.get("SENTRY_DSN")


def calculate_content_hash(content: str) -> str:
    """Return a stable SHA256 hash for the provided text content."""
    if content is None:
        content = ""
    if not isinstance(content, str):
        content = str(content)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
    """Reranking orchestrator that applies MMR directly over provided documents.

    This function acts as a zero-cost drop-in replacement for Cohere.
    """
    # Track whether caller passed LangChain Documents (for faithful return type)
    input_is_langchain = bool(documents) and hasattr(documents[0], "page_content") and hasattr(documents[0], "metadata")
    # Normalize input docs in case caller passes LangChain Documents
    normalized_input_docs: List[Dict[str, Any]] = [_normalize_doc_input(d) for d in (documents or [])]
    
    # Build master map of id -> doc from the initial input documents
    all_docs_map: Dict[str, Dict[str, Any]] = {}
    for d in normalized_input_docs:
        all_docs_map.setdefault(_get_doc_id(d), d)

    # Use existing similarity score as the relevance score for MMR
    fused_scores: Dict[str, float] = {}
    for doc in normalized_input_docs:
        doc_id = _get_doc_id(doc)
        score = doc.get("similarity") or 0.0
        fused_scores[doc_id] = float(score)

    # --- Fallback: Return original documents if no scores are available ---
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
        logging.warning(f"Query embedding failed for MMR; returning top by similarity only: {e}")
        # If embedding fails, return top-7 by similarity score without MMR
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
genai = None
genai_types = None
try:
    from google import genai as _genai_mod  # Preferred for google-genai>=1.0
    genai = _genai_mod
except Exception:
    try:
        import google.generativeai as _legacy_genai  # Backwards compatibility
        genai = _legacy_genai
    except Exception:
        genai = None  # Optional dependency

if genai is not None:
    try:
        from google.genai import types as genai_types  # type: ignore[attr-defined]
    except Exception:
        try:
            genai_types = getattr(genai, "types", None)  # legacy packages
        except Exception:
            genai_types = None

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
_GENAI_EMBED_WARNING_EMITTED = False
_OPENAI_DISABLED = False
_OPENAI_DISABLED_REASON = ""

_MATCH_DOCUMENTS_RPC_STYLE: Optional[str] = None
_MATCH_DOCUMENTS_RPC_STYLE_HINT = os.environ.get("MATCH_DOCUMENTS_RPC_STYLE", "").strip().lower()
if _MATCH_DOCUMENTS_RPC_STYLE_HINT in {"legacy", "modern"}:
    _MATCH_DOCUMENTS_RPC_STYLE = _MATCH_DOCUMENTS_RPC_STYLE_HINT


def _build_embed_config(task_type: str | None = "retrieval_query"):
    """Return google-genai EmbedContentConfig when available."""
    if not task_type or genai_types is None:
        return None
    for key in ("taskType", "task_type"):
        try:
            return genai_types.EmbedContentConfig(**{key: task_type})  # type: ignore[arg-type]
        except Exception:
            continue
    return None


def _handle_openai_embedding_error(error: Exception) -> None:
    """Set a sticky flag when OpenAI fallback is unusable."""
    global _OPENAI_DISABLED, _OPENAI_DISABLED_REASON
    message = str(error) if error else ""
    # Prefer stricter matching on known status codes when available
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    http_status = getattr(error, "status_code", None)
    status = status_code or http_status

    if status == 429 or "insufficient_quota" in message.lower() or "exceeded your current quota" in message.lower():
        _OPENAI_DISABLED = True
        _OPENAI_DISABLED_REASON = "insufficient_quota"
        logging.error(
            "OpenAI embedding fallback disabled after insufficient quota error. "
            "Provide a funded OpenAI account or configure GEMINI_API_KEY/GOOGLE_API_KEY to avoid this fallback. "
            "Set DISABLE_OPENAI_FALLBACK=1 to suppress OpenAI usage altogether."
        )
    else:
        logging.warning(f"OpenAI embedding fallback failed: {error}")


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
    global _GENAI_CLIENT, _GENAI_EMBED_WARNING_EMITTED
    try:
        if genai is None:
            return
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            if not _GENAI_EMBED_WARNING_EMITTED:
                logging.warning(
                    "Google GenAI embeddings unavailable: set GEMINI_API_KEY or GOOGLE_API_KEY to avoid falling back to OpenAI."
                )
                _GENAI_EMBED_WARNING_EMITTED = True
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
        if not _GENAI_EMBED_WARNING_EMITTED:
            logging.warning(
                "Failed to initialize Google GenAI embeddings; will attempt OpenAI fallback if enabled."
            )
            _GENAI_EMBED_WARNING_EMITTED = True


def _ensure_openai():
    """Best-effort initialize OpenAI client for embeddings fallback."""
    global _OPENAI_CLIENT
    if DISABLE_OPENAI_FALLBACK or _OPENAI_DISABLED:
        return
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


def _is_supabase_signature_mismatch(exc: Exception) -> bool:
    """Return True if the Supabase RPC error looks like a signature mismatch."""
    try:
        fragments: List[str] = [str(exc)]
    except Exception:
        fragments = []

    for attr in ("message", "hint", "details"):
        value = getattr(exc, attr, None)
        if value:
            text = str(value)
            if text not in fragments:
                fragments.append(text)

    blob = " ".join(fragments)
    if "match_documents(filter" in blob:
        return True
    if "No function matches the given name" in blob and "match_documents" in blob:
        return True

    code = getattr(exc, "code", "")
    if code == "PGRST202":
        return True
    if PostgrestAPIError is not None and isinstance(exc, PostgrestAPIError):
        if getattr(exc, "code", "") == "PGRST202":
            return True

    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    response = getattr(exc, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status == 404:
            return True

    return False


def _invoke_match_documents_rpc(
    supabase_client,
    *,
    query_embedding: List[float],
    match_threshold: float,
    match_count: int,
    selected_class: Optional[str],
):
    """Call match_documents with adaptive payloads, caching the working signature."""
    global _MATCH_DOCUMENTS_RPC_STYLE

    style_preference: List[str] = []
    if _MATCH_DOCUMENTS_RPC_STYLE:
        style_preference.append(_MATCH_DOCUMENTS_RPC_STYLE)
    elif _MATCH_DOCUMENTS_RPC_STYLE_HINT:
        style_preference.append(_MATCH_DOCUMENTS_RPC_STYLE_HINT)
    else:
        style_preference.append("legacy")

    if "modern" not in style_preference:
        style_preference.append("modern")
    if "legacy" not in style_preference:
        style_preference.append("legacy")

    last_exc: Exception | None = None

    for style in style_preference:
        class_filter = (selected_class or "").strip() or None
        if style == "modern":
            payload = {
                "query_embedding": query_embedding,
                "match_threshold": float(match_threshold),
                "match_count": int(match_count),
            }
            if class_filter:
                payload["filter_class"] = class_filter
        else:  # legacy JSON filter signature
            legacy_filter: Dict[str, Any] = {"match_threshold": float(match_threshold)}
            if class_filter:
                legacy_filter["class_name"] = class_filter
            payload = {
                "query_embedding": query_embedding,
                "match_count": int(match_count),
                "filter": legacy_filter,
            }

        try:
            response = supabase_client.rpc("match_documents", payload).execute()
            if _MATCH_DOCUMENTS_RPC_STYLE != style:
                _MATCH_DOCUMENTS_RPC_STYLE = style
                logging.info("Supabase match_documents RPC signature detected: %s", style)
            return response
        except Exception as exc:
            last_exc = exc
            if _is_supabase_signature_mismatch(exc):
                logging.debug("match_documents payload style '%s' rejected: %s", style, exc)
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unable to execute match_documents RPC with available payload styles")


def embed_query(text: str, model: str | None = None) -> list[float]:
    """Embed a single query string with robust fallbacks (Gemini -> OpenAI)."""
    _ensure_genai()
    # Try the provided model first, then sensible fallbacks
    primary_model = model or EMBEDDING_MODEL_NAME
    candidate_models: list[str] = []
    for m in (primary_model, "text-embedding-004", "models/embedding-001"):
        if m and m not in candidate_models:
            candidate_models.append(m)

    gemini_attempted = False
    last_gemini_error: Exception | None = None

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
                if hasattr(first, "values") and isinstance(getattr(first, "values"), list):
                    return list(getattr(first, "values"))  # type: ignore[arg-type]
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
            config = _build_embed_config("retrieval_query")
            kwargs = {
                "model": embedding_model,
                "contents": [text],
            }
            if config is not None:
                kwargs["config"] = config
            try:
                gemini_attempted = True
                result = _GENAI_CLIENT.models.embed_content(**kwargs)
                result_vec = _extract_single(result)  # existing line

                # Validate dimension matches database constraint
                if len(result_vec) != EXPECTED_EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch: model {embedding_model} returned "
                        f"{len(result_vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                        f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                    )
                return result_vec
            except TypeError as err:
                last_gemini_error = err
                # Older signature that accepts 'content='
                try:
                    gemini_attempted = True
                    result = _GENAI_CLIENT.models.embed_content(
                        model=embedding_model,
                        content=text,
                        task_type="retrieval_query",
                    )
                    result_vec = _extract_single(result)  # existing line

                    # Validate dimension matches database constraint
                    if len(result_vec) != EXPECTED_EMBEDDING_DIM:
                        raise ValueError(
                            f"Embedding dimension mismatch: model {embedding_model} returned "
                            f"{len(result_vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                            f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                        )
                    return result_vec
                except Exception as err:
                    last_gemini_error = err
                    pass
            except Exception as err:
                # Try alternative API below
                last_gemini_error = err

            # Alternative new API: generate_content_embeddings
            try:
                gce = getattr(_GENAI_CLIENT.models, "generate_content_embeddings", None)
                if gce is not None:
                    gemini_attempted = True
                    result = gce(
                        model=embedding_model,
                        requests=[{"content": {"text": text}}],
                    )
                    result_vec = _extract_single(result)  # existing line

                    # Validate dimension matches database constraint
                    if len(result_vec) != EXPECTED_EMBEDDING_DIM:
                        raise ValueError(
                            f"Embedding dimension mismatch: model {embedding_model} returned "
                            f"{len(result_vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                            f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                        )
                    return result_vec
            except Exception as err:
                last_gemini_error = err
                pass

        # Fallback: module-level API (older libraries or tests)
        try:
            if hasattr(genai, "embed_content"):
                gemini_attempted = True
                result = genai.embed_content(
                    model=embedding_model,
                    content=text,
                    task_type="retrieval_query",
                )
                result_vec = _extract_single(result)  # existing line

                # Validate dimension matches database constraint
                if len(result_vec) != EXPECTED_EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch: model {embedding_model} returned "
                        f"{len(result_vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                        f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                    )
                return result_vec
        except Exception as err:
            last_gemini_error = err
            pass
    if gemini_attempted:
        if last_gemini_error:
            logging.warning(
                "Gemini embedding attempts failed for all candidate models; last error: %s",
                last_gemini_error,
            )
    else:
        logging.warning(
            "Gemini embeddings were not attempted. Verify GEMINI_API_KEY or GOOGLE_API_KEY configuration to avoid OpenAI fallback."
        )

    # OpenAI fallback
    if DISABLE_OPENAI_FALLBACK:
        logging.debug("Skipping OpenAI embedding fallback because DISABLE_OPENAI_FALLBACK is set.")
    else:
        _ensure_openai()
        if _OPENAI_CLIENT is not None and not _OPENAI_DISABLED:
            try:
                openai_model = os.environ.get("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
                resp = _OPENAI_CLIENT.embeddings.create(
                    model=openai_model,
                    input=text,
                )
                data = getattr(resp, "data", None) or []
                if data and hasattr(data[0], "embedding"):
                    result_vec = data[0].embedding  # type: ignore[assignment]

                    # Validate dimension matches database constraint
                    if len(result_vec) != EXPECTED_EMBEDDING_DIM:
                        raise ValueError(
                            f"Embedding dimension mismatch: model {openai_model} returned "
                            f"{len(result_vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                            f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                        )

                    return result_vec
            except Exception as e:
                _handle_openai_embedding_error(e)
        elif _OPENAI_DISABLED:
            logging.debug(
                "Skipping OpenAI embedding fallback because it was previously disabled (%s).",
                _OPENAI_DISABLED_REASON or "unknown reason",
            )

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

    gemini_attempted = False
    last_gemini_error: Exception | None = None

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
                    elif hasattr(item, "values") and isinstance(getattr(item, "values"), list):
                        out.append(list(getattr(item, "values")))  # type: ignore[list-item]
                if out:
                    return out
        raise RuntimeError(f"Unexpected batch embed_content result shape: {type(result)}")

    # Try each candidate model across known SDK call styles
    for embedding_model in candidate_models:
        # Try modern client API first (prefer 'contents' per new SDK)
        if _GENAI_CLIENT is not None and getattr(_GENAI_CLIENT, "models", None) is not None:
            config = _build_embed_config("retrieval_query")
            kwargs = {
                "model": embedding_model,
                "contents": texts,
            }
            if config is not None:
                kwargs["config"] = config
            try:
                gemini_attempted = True
                result = _GENAI_CLIENT.models.embed_content(**kwargs)
                vectors = _extract_batch(result)

                # Validate dimension matches database constraint
                for vec in vectors:
                    if len(vec) != EXPECTED_EMBEDDING_DIM:
                        raise ValueError(
                            f"Embedding dimension mismatch: model {embedding_model} returned "
                            f"{len(vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                            f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                        )
                return vectors
            except TypeError as err:
                last_gemini_error = err
                try:
                    gemini_attempted = True
                    result = _GENAI_CLIENT.models.embed_content(
                        model=embedding_model,
                        content=texts,
                        task_type="retrieval_query",
                    )
                    vectors = _extract_batch(result)

                    # Validate dimension matches database constraint
                    for vec in vectors:
                        if len(vec) != EXPECTED_EMBEDDING_DIM:
                            raise ValueError(
                                f"Embedding dimension mismatch: model {embedding_model} returned "
                                f"{len(vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                                f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                            )
                    return vectors
                except Exception as err:
                    last_gemini_error = err
                    pass
            except Exception as err:
                last_gemini_error = err
                pass

            # Alternative new API: generate_content_embeddings
            try:
                gce = getattr(_GENAI_CLIENT.models, "generate_content_embeddings", None)
                if gce is not None:
                    gemini_attempted = True
                    result = gce(
                        model=embedding_model,
                        requests=[{"content": {"text": t}} for t in texts],
                    )
                    vectors = _extract_batch(result)

                    # Validate dimension matches database constraint
                    for vec in vectors:
                        if len(vec) != EXPECTED_EMBEDDING_DIM:
                            raise ValueError(
                                f"Embedding dimension mismatch: model {embedding_model} returned "
                                f"{len(vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                                f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                            )
                    return vectors
            except Exception as err:
                last_gemini_error = err
                pass

        # Fallback: module-level API (older libraries or tests)
        try:
            if hasattr(genai, "embed_content"):
                gemini_attempted = True
                result = genai.embed_content(
                    model=embedding_model,
                    content=texts,
                    task_type="retrieval_query",
                )
                vectors = _extract_batch(result)

                # Validate dimension matches database constraint
                for vec in vectors:
                    if len(vec) != EXPECTED_EMBEDDING_DIM:
                        raise ValueError(
                            f"Embedding dimension mismatch: model {embedding_model} returned "
                            f"{len(vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                            f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                        )
                return vectors
        except Exception as err:
            last_gemini_error = err
            pass
    if gemini_attempted:
        if last_gemini_error:
            logging.warning(
                "Gemini batch embedding attempts failed for all candidate models; last error: %s",
                last_gemini_error,
            )
    else:
        logging.warning(
            "Gemini batch embeddings were not attempted. Verify GEMINI_API_KEY or GOOGLE_API_KEY configuration to avoid OpenAI fallback."
        )

    # OpenAI batch fallback
    if DISABLE_OPENAI_FALLBACK:
        logging.debug("Skipping OpenAI batch embedding fallback because DISABLE_OPENAI_FALLBACK is set.")
    else:
        _ensure_openai()
        if _OPENAI_CLIENT is not None and not _OPENAI_DISABLED:
            try:
                openai_model = os.environ.get("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
                resp = _OPENAI_CLIENT.embeddings.create(
                    model=openai_model,
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
                        # Validate dimension matches database constraint
                        for vec in vectors:
                            if len(vec) != EXPECTED_EMBEDDING_DIM:
                                raise ValueError(
                                    f"Embedding dimension mismatch: model {openai_model} returned "
                                    f"{len(vec)}-dimensional vector, but database expects {EXPECTED_EMBEDDING_DIM}. "
                                    f"Update database/match_documents.sql or change EMBEDDING_MODEL_NAME."
                                )

                        return vectors
            except Exception as e:
                _handle_openai_embedding_error(e)
        elif _OPENAI_DISABLED:
            logging.debug(
                "Skipping OpenAI batch embedding fallback because it was previously disabled (%s).",
                _OPENAI_DISABLED_REASON or "unknown reason",
            )

    # Ultimate fallback: individual calls
    logging.warning("Batch embedding failed, falling back to individual calls")
    return [embed_query(text, model) for text in texts]


def retrieve_rag_documents(
    query: str,
    selected_class: str | None = None,
    match_count: int = 20,
    match_threshold: float = 0.7,
    query_embedding: List[float] | None = None,
    enable_hybrid: bool = True,
    rpc_function_name: str = "match_documents_hybrid",
) -> list[dict]:
    """Retrieve documents via Supabase using Hybrid RRF search."""
    start_time = time.time()

    supabase = _get_supabase_client()

    try:
        if query_embedding is None:
            embed_start = time.time()
            try:
                query_embedding = embed_query(query)
                embed_duration_ms = (time.time() - embed_start) * 1000

                # Track embedding costs (approximate)
                # text-embedding-004: $0.00001 per 1K tokens
                estimated_tokens = len(query.split()) * 1.3  # rough estimate
                estimated_cost = (estimated_tokens / 1000) * 0.00001

                logging.debug(
                    f"EMBEDDING_METRICS | duration_ms={embed_duration_ms:.2f} "
                    f"tokens~{estimated_tokens:.0f} cost_usd~{estimated_cost:.6f}"
                )
            except Exception:
                return []

        # Hybrid Path (RRF)
        payload = {
            "query_embedding": query_embedding,
            "query_text": query,
            "match_count": int(match_count),
            "rrf_k": 60,
        }
        if selected_class:
            payload["filter_class"] = selected_class

        try:
            response = supabase.rpc(rpc_function_name, payload).execute()
            results = getattr(response, "data", None) or []

            duration_ms = (time.time() - start_time) * 1000

            # Log hybrid DB performance
            logging.info(
                f"HYBRID_SEARCH | duration_ms={duration_ms:.2f} "
                f"results={len(results)} "
                f"class_filter={selected_class or 'none'} "
                f"rpc_function={rpc_function_name}"
            )
        except Exception as rpc_exc:
            logging.error("Supabase match_documents_hybrid RPC failed: %s", rpc_exc)
            raise

        # Alert on slow queries (>1s)
        if duration_ms > 1000 and SENTRY_DSN and sentry_sdk:
            sentry_sdk.capture_message(
                f"Slow hybrid DB query: {duration_ms:.0f}ms",
                level="warning",
            )

        return results

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(
            f"VECTOR_DB_ERROR | duration_ms={duration_ms:.2f} error={type(e).__name__}"
        )
        if SENTRY_DSN and sentry_sdk:
            sentry_sdk.capture_exception(e)
        raise


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