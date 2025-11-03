from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from src.shared.utils import (
    EMBEDDING_MODEL_NAME,
    cohere_rerank,
    retrieve_rag_documents,
    retrieve_rag_documents_keyword_fallback,
)
try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - new SDK path
    from google import genai  # type: ignore[import]
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
import time

# --- Sentry Integration (Optional) ---
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    HAS_SENTRY = True
except ImportError:
    sentry_sdk = None
    HAS_SENTRY = False


def sanitize_sentry_event(event, hint):
    """Remove sensitive data before sending to Sentry."""
    if "request" in event and "env" in event["request"]:
        sensitive_keys = [
            "GEMINI_API_KEY",
            "SUPABASE_KEY",
            "SECRET_API_KEY",
            "OPENAI_API_KEY",
            "COACH_PASSWORD",
            "SENTRY_DSN",
        ]
        for key in sensitive_keys:
            if key in event["request"]["env"]:
                event["request"]["env"][key] = "[REDACTED]"

    if "request" in event and "headers" in event["request"]:
        if "x-api-key" in event["request"]["headers"]:
            event["request"]["headers"]["x-api-key"] = "[REDACTED]"
        if "authorization" in event["request"]["headers"]:
            event["request"]["headers"]["authorization"] = "[REDACTED]"

    if "request" in event:
        if "query_string" in event["request"]:
            event["request"]["query_string"] = "[SANITIZED]"
        if "data" in event["request"] and isinstance(event["request"]["data"], dict):
            if "api_key" in event["request"]["data"]:
                event["request"]["data"]["api_key"] = "[REDACTED]"

    return event


# Initialize Sentry if DSN is provided
SENTRY_DSN = os.getenv("SENTRY_DSN")
SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "production")
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))

if SENTRY_DSN and HAS_SENTRY:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=SENTRY_ENVIRONMENT,
        traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            LoggingIntegration(level=logging.INFO, event_level=logging.ERROR),
        ],
        before_send=sanitize_sentry_event,
        profiles_sample_rate=0.1,
    )
    logging.info("Sentry monitoring initialized")
elif SENTRY_DSN and not HAS_SENTRY:
    logging.warning("SENTRY_DSN provided but sentry-sdk not installed. Install with: pip install sentry-sdk[fastapi]")

# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Performance Metrics (In-Memory) ---
REQUEST_COUNT: Dict[str, int] = {}
REQUEST_DURATION: Dict[str, List[float]] = {}
EMBEDDING_COSTS = {"total_tokens": 0, "estimated_cost_usd": 0.0}
VECTOR_DB_METRICS = {"query_count": 0, "avg_latency_ms": 0.0, "cache_hits": 0}


@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """Track request performance and log slow endpoints."""
    start_time = time.time()
    path = request.url.path

    # Increment request count per path
    REQUEST_COUNT[path] = REQUEST_COUNT.get(path, 0) + 1

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        # Track request duration per path
        if path not in REQUEST_DURATION:
            REQUEST_DURATION[path] = []
        REQUEST_DURATION[path].append(duration_ms)

        # Log slow requests (>2 seconds)
        if duration_ms > 2000:
            logging.warning(
                "SLOW_REQUEST | path=%s duration_ms=%.2f status=%s",
                path,
                duration_ms,
                response.status_code,
            )

            if SENTRY_DSN and HAS_SENTRY:
                with sentry_sdk.configure_scope() as scope:
                    scope.set_context(
                        "performance",
                        {
                            "endpoint": path,
                            "duration_ms": duration_ms,
                            "threshold_ms": 2000,
                        },
                    )
                    sentry_sdk.capture_message(
                        f"Slow endpoint: {path}",
                        level="warning",
                    )

        return response
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(
            "REQUEST_ERROR | path=%s duration_ms=%.2f error=%s",
            path,
            duration_ms,
            type(exc).__name__,
        )
        raise


# --- CORS Setup (Robust, Environment-based) ---
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

# Fallback defaults when env var not provided (Quick Test / local dev)
if not allowed_origins:
    allowed_origins = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize clients ---
_GENAI_CLIENT = None
_api_key = os.getenv("GEMINI_API_KEY")
if _api_key:
    try:
        _GENAI_CLIENT = genai.Client(api_key=_api_key)
    except Exception:
        _GENAI_CLIENT = None

# Optional Supabase client for health checks
supabase = None
try:
    from src.shared.utils import _get_supabase_client
    supabase = _get_supabase_client()
except Exception:
    supabase = None

EMBEDDING_MODEL = EMBEDDING_MODEL_NAME

# --- API Key Authentication Setup ---
api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Robust API key validation with immediate rejection of invalid requests."""
    expected = os.getenv("SECRET_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="API key not configured")
    
    api_key = api_key.strip() if api_key else ""
    
    if len(api_key) < 8:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# --- Graceful Fallback Message ---
NO_DOCUMENTS_ANSWER = (
    "I couldn't find relevant materials. This could be because:\n"
    "- The content hasn't been uploaded yet\n"
    "- Try rephrasing your question\n"
    "- Contact alanayalag@gmail.com if this persists"
)

# --- Logging Setup ---
logger = logging.getLogger("api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - api - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _check_db_status() -> Dict[str, Optional[bool]]:
    """Perform a trivial Supabase query to confirm connectivity."""
    if supabase is None:
        return {"connected": False, "ok": False}
    try:
        _ = supabase.table("documents").select("id").limit(1).execute()
        return {"connected": True, "ok": True}
    except Exception:
        return {"connected": True, "ok": False}


def _get_rag_params() -> Dict[str, object]:
    """Read RAG tuning parameters from environment with safe defaults."""
    try:
        match_threshold = float(os.getenv("RAG_MATCH_THRESHOLD", "0.5"))
    except Exception:
        match_threshold = 0.5
    try:
        match_count = int(os.getenv("RAG_MATCH_COUNT", "8"))
    except Exception:
        match_count = 8
    relaxed_threshold = min(match_threshold, 0.3)
    relaxed_count = max(match_count, 10)
    return {
        "match_threshold": match_threshold,
        "match_count": match_count,
        "relaxed_threshold": relaxed_threshold,
        "relaxed_count": relaxed_count,
    }


class Message(BaseModel):
    role: str
    content: str
    sources: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    selectedClass: Optional[str] = None
    persona: str = "balanced"


class RAGRequest(BaseModel):
    query: str
    selectedClass: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Track application start time for uptime metrics."""
    app.state.start_time = time.time()


@app.get("/")
@limiter.limit("100/minute")
async def health_check(request: Request):
    """Enhanced health check for monitoring."""
    db_status = _check_db_status()
    return {
        "status": "healthy",
        "service": "AI Tutor API",
        "version": "1.0.0-beta",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "model_loaded": _GENAI_CLIENT is not None,
        "endpoints": {
            "chat": "/api/chat",
            "rag_search": "/api/rag-search",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
@limiter.limit("100/minute")
async def health(request: Request):
    """Dedicated health endpoint for uptime monitoring."""
    db_status = _check_db_status()
    return {"status": "ok", "database": db_status}


@app.get("/metrics")
async def get_metrics(api_key: str = Depends(get_api_key)):
    """Expose performance metrics for monitoring (protected by API key)."""
    metrics = {
        "request_counts": REQUEST_COUNT,
        "avg_response_times_ms": {
            path: (sum(durations) / len(durations)) if durations else 0
            for path, durations in REQUEST_DURATION.items()
        },
        "embedding_costs": EMBEDDING_COSTS,
        "vector_db_metrics": VECTOR_DB_METRICS,
        "uptime_seconds": time.time() - getattr(app.state, "start_time", time.time()),
    }
    return metrics


@app.post("/api/rag-search")
@limiter.limit("20/minute")
async def rag_search(request: Request, payload: RAGRequest, api_key: str = Depends(get_api_key)):
    _t0 = time.time()
    try:
        params = _get_rag_params()
        documents = retrieve_rag_documents(
            query=payload.query,
            selected_class=payload.selectedClass,
            match_count=params["match_count"],
            match_threshold=params["match_threshold"],
        )
        documents = cohere_rerank(payload.query, documents)
        
        if not documents:
            documents = retrieve_rag_documents(
                query=payload.query,
                selected_class=None,
                match_count=params["relaxed_count"],
                match_threshold=params["relaxed_threshold"],
            )
            documents = cohere_rerank(payload.query, documents)
        
        if documents:
            avg_sim = sum([d.get("similarity", 0) or 0 for d in documents]) / max(len(documents), 1)
            logger.info(
                "RAG_SEARCH metrics | query_len=%s docs=%s avg_similarity=%.4f latency_ms=%s",
                len(payload.query or ""),
                len(documents),
                avg_sim,
                int((time.time() - _t0) * 1000),
            )
        
        if not documents:
            kw_docs = retrieve_rag_documents_keyword_fallback(
                query=payload.query,
                selected_class=payload.selectedClass,
                limit=params["relaxed_count"],
            )
            if not kw_docs and payload.selectedClass:
                kw_docs = retrieve_rag_documents_keyword_fallback(
                    query=payload.query,
                    selected_class=None,
                    limit=params["relaxed_count"],
                )
            documents = kw_docs
        
        if not documents:
            logger.warning(
                "RAG_SEARCH failure | NO_DOCUMENTS_ANSWER | query_len=%s latency_ms=%s",
                len(payload.query or ""),
                int((time.time() - _t0) * 1000),
            )
            return {
                "answer": NO_DOCUMENTS_ANSWER,
                "sources": [],
            }
        return {"documents": documents}
    except Exception as e:
        logger.exception("RAG_SEARCH error | latency_ms=%s", int((time.time() - _t0) * 1000))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_stream(request: Request, payload: ChatRequest, api_key: str = Depends(get_api_key)):
    _t0 = time.time()
    try:
        user_messages = [m for m in payload.messages if (m.role or "").lower() == "user"]
        last_query = user_messages[-1].content if user_messages else ""

        params = _get_rag_params()
        documents = retrieve_rag_documents(
            query=last_query,
            selected_class=payload.selectedClass,
            match_count=params["match_count"],
            match_threshold=params["match_threshold"],
        )
        documents = cohere_rerank(last_query, documents)
        
        if not documents:
            documents = retrieve_rag_documents(
                query=last_query,
                selected_class=None,
                match_count=params["relaxed_count"],
                match_threshold=params["relaxed_threshold"],
            )
            documents = cohere_rerank(last_query, documents)
        
        if not documents:
            kw_docs = retrieve_rag_documents_keyword_fallback(
                query=last_query,
                selected_class=payload.selectedClass,
                limit=params["relaxed_count"],
            )
            if not kw_docs and payload.selectedClass:
                kw_docs = retrieve_rag_documents_keyword_fallback(
                    query=last_query,
                    selected_class=None,
                    limit=params["relaxed_count"],
                )
            documents = kw_docs
        
        if documents:
            avg_sim = sum([d.get("similarity", 0) or 0 for d in documents]) / max(len(documents), 1)
        else:
            avg_sim = 0.0
        logger.info(
            "CHAT metrics | query_len=%s docs=%s avg_similarity=%.4f latency_ms=%s",
            len(last_query or ""),
            len(documents or []),
            avg_sim,
            int((time.time() - _t0) * 1000),
        )
        
        context = "\n\n".join([
            f"Source {i+1}:\nClass: {doc.get('class_name', 'N/A')}\n{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        ]) if documents else "No relevant course materials found."

        personas = {
            "study": "You are a study buddy helping review material...",
            "professor": "You are an experienced professor teaching...",
            "socratic": "You use the Socratic method...",
            "balanced": "You are a balanced tutor..."
        }
        
        if documents:
            rules = (
                "- ONLY use information from the provided course materials\n"
                "- If information is not in the materials, say so\n"
                "- Cite sources when referencing specific content\n"
                "- Be conversational but accurate"
            )
        else:
            rules = (
                "- No course materials were found for this query.\n"
                "- Answer based on your general knowledge and reasoning.\n"
                "- Preface the answer with a brief note that course materials were unavailable.\n"
                "- Be concise, accurate, and helpful"
            )

        system_prompt = f"""{personas.get(payload.persona, personas['balanced'])}

COURSE MATERIALS:
{context}

RULES:
{rules}"""
        
        def _to_genai_contents(chat_messages: List[Dict[str, str]]):
            contents: List[Dict[str, object]] = []
            for m in chat_messages:
                role_raw = (m.get("role") or "").lower()
                if role_raw == "system":
                    continue
                role = "user" if role_raw == "user" else "model" if role_raw == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": m.get("content", "")}],
                })
            return contents
        
        clean_messages: List[Dict[str, str]] = [
            {"role": m.role, "content": m.content} for m in payload.messages
        ]
        genai_contents = _to_genai_contents(clean_messages)
        
        async def generate():
            sources_payload = {
                "sources": [
                    {
                        "content": doc.get("content", ""),
                        "metadata": {
                            "class_name": doc.get("class_name"),
                            "section": doc.get("section"),
                            "title": doc.get("title"),
                            "url": doc.get("url")
                        },
                        "similarity": doc.get("similarity", 0)
                    }
                    for doc in (documents or [])
                ]
            }
            if not documents:
                logger.warning(
                    "CHAT fallback | proceeding without materials | query_len=%s latency_ms=%s",
                    len(last_query or ""),
                    int((time.time() - _t0) * 1000),
                )
            yield f"data: {json.dumps(sources_payload)}\n\n"
            
            if _GENAI_CLIENT is None:
                raise RuntimeError("Gemini client is not initialized. Check GEMINI_API_KEY.")

            try:
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    system_instruction=system_prompt,
                    client=_GENAI_CLIENT,
                )

                response = model.generate_content(
                    contents=genai_contents,
                    stream=True,
                )

                for chunk in response:
                    chunk_text = getattr(chunk, "text", None)

                    if not chunk_text and getattr(chunk, "candidates", None):
                        parts: List[str] = []
                        for candidate in chunk.candidates:
                            content = getattr(candidate, "content", None)
                            if not content:
                                continue
                            for part in getattr(content, "parts", []) or []:
                                text = getattr(part, "text", None)
                                if text:
                                    parts.append(text)
                        chunk_text = "".join(parts) if parts else None

                    if chunk_text:
                        data = {
                            "choices": [{
                                "delta": {"content": chunk_text}
                            }]
                        }
                        yield f"data: {json.dumps(data)}\n\n"
            except Exception as stream_err:
                err_payload = {"error": str(stream_err)}
                yield f"data: {json.dumps(err_payload)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )
        
    except Exception as e:
        logger.exception("CHAT error | latency_ms=%s", int((time.time() - _t0) * 1000))
        raise HTTPException(status_code=500, detail=str(e))