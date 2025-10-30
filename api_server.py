from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from src.shared.utils import EMBEDDING_MODEL_NAME, cohere_rerank, retrieve_rag_documents
from google import genai
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
import time

# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS Setup (Robust, Environment-based) ---
# Get Lovable frontend URL(s) from environment. Comma-separated.
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

# Fallback defaults when env var not provided (Quick Test / local dev)
if not allowed_origins:
    allowed_origins = [
        # Lovable frontend
        "https://93a4e185-b263-4e0d-83e0-9cf4863ef461.lovableproject.com",
        # Local development
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
# Uses GEMINI_API_KEY environment variable
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

# IMPORTANT: Keep this aligned with src/refinery/embedding.py
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME

# --- API Key Authentication Setup ---
api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)

def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Robust API key validation with immediate rejection of invalid requests.
    
    Returns:
        str: Valid API key
        
    Raises:
        HTTPException: 500 if SECRET_API_KEY not configured
        HTTPException: 403 if API key is invalid
    """
    expected = os.getenv("SECRET_API_KEY")
    if not expected:
        # Service misconfiguration; reject until configured
        raise HTTPException(status_code=500, detail="API key not configured")
    
    # Handle whitespace issues (common with copy/paste)
    api_key = api_key.strip() if api_key else ""
    
    # Validate minimum length to prevent empty/trivial keys
    if len(api_key) < 8:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    if api_key != expected:
        # Use 403 Forbidden since the key is likely just wrong
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# --- Graceful Fallback Message when no documents are found ---
NO_DOCUMENTS_ANSWER = (
    "I couldn't find relevant materials. This could be because:\n"
    "- The content hasn't been uploaded yet\n"
    "- Try rephrasing your question\n"
    "- Contact alanayalag@gmail.com if this persists"
)

# --- Basic API logging setup (stdout). Avoid duplicate handlers.
logger = logging.getLogger("api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - api - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _check_db_status() -> Dict[str, Optional[bool]]:
    """Perform a trivial Supabase query to confirm connectivity.
    Returns a dict with keys 'connected' and 'ok'.
    """
    if supabase is None:
        return {"connected": False, "ok": False}
    try:
        _ = supabase.from_("documents").select("id").limit(1).execute()
        return {"connected": True, "ok": True}
    except Exception:
        return {"connected": True, "ok": False}

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    selectedClass: Optional[str] = None
    persona: str = "balanced"

class RAGRequest(BaseModel):
    query: str
    selectedClass: Optional[str] = None

@app.get("/")
async def health_check():
    """Enhanced health check for Render monitoring"""
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
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Dedicated health endpoint for uptime monitoring"""
    db_status = _check_db_status()
    return {"status": "ok", "database": db_status}

@app.post("/api/rag-search")
async def rag_search(payload: RAGRequest, api_key: str = Depends(get_api_key)):
    _t0 = time.time()
    try:
        # Standardized retrieval via shared utility (Supabase RPC + Gemini embeddings)
        documents = retrieve_rag_documents(
            query=payload.query,
            selected_class=payload.selectedClass,
            match_count=5,
            match_threshold=0.7,
        )
        # Re-ranking using the zero-cost RRF/MMR implementation
        documents = cohere_rerank(payload.query, documents)
        # Telemetry: average similarity
        if documents:
            avg_sim = sum([d.get("similarity", 0) or 0 for d in documents]) / max(len(documents), 1)
            logger.info(
                "RAG_SEARCH metrics | query_len=%s docs=%s avg_similarity=%.4f latency_ms=%s",
                len(payload.query or ""),
                len(documents),
                avg_sim,
                int((time.time() - _t0) * 1000),
            )
        # Graceful fallback when no documents are found
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
        # Get RAG context
        user_messages = [m for m in payload.messages if m.get("role") == "user"]
        last_query = user_messages[-1]["content"] if user_messages else ""

        # Retrieve documents directly
        documents = retrieve_rag_documents(
            query=last_query,
            selected_class=payload.selectedClass,
            match_count=5,
            match_threshold=0.7,
        )
        # Re-ranking using the zero-cost RRF/MMR implementation
        documents = cohere_rerank(last_query, documents)
        # Telemetry: average similarity and request metrics
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
        
        # Build context
        context = "\n\n".join([
            f"Source {i+1}:\nClass: {doc.get('class_name', 'N/A')}\n{doc.get('content', '')}"
            for i, doc in enumerate(documents)
        ]) if documents else "No relevant course materials found."
        
        # System prompt
        personas = {
            "study": "You are a study buddy helping review material...",
            "professor": "You are an experienced professor teaching...",
            "socratic": "You use the Socratic method...",
            "balanced": "You are a balanced tutor..."
        }
        
        system_prompt = f"""{personas.get(payload.persona, personas['balanced'])}

COURSE MATERIALS:
{context}

RULES:
- ONLY use information from the provided course materials
- If information is not in the materials, say so
- Cite sources when referencing specific content
- Be conversational but accurate"""
        
        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        for msg in payload.messages:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Stream response
        async def generate():
            # Send sources first
            if documents:
                sources_data = {
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
                        for doc in documents
                    ]
                }
                yield f"data: {json.dumps(sources_data)}\n\n"
            else:
                # Graceful fallback when no documents are found
                logger.warning(
                    "CHAT failure | NO_DOCUMENTS_ANSWER | query_len=%s latency_ms=%s",
                    len(last_query or ""),
                    int((time.time() - _t0) * 1000),
                )
                yield f"data: {json.dumps({'sources': []})}\n\n"
                yield f"data: {json.dumps({'choices': [{'delta': {'content': NO_DOCUMENTS_ANSWER}}]})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Stream AI response
            if _GENAI_CLIENT is None:
                raise RuntimeError("Gemini client is not initialized. Check GEMINI_API_KEY.")

            response = _GENAI_CLIENT.models.generate_content(
                model="gemini-2.5-flash",
                contents=[m["content"] for m in messages],
                stream=True,
            )
            
            for chunk in response:
                if chunk.text:
                    data = {
                        "choices": [{
                            "delta": {"content": chunk.text}
                        }]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        logger.exception("CHAT error | latency_ms=%s", int((time.time() - _t0) * 1000))
        raise HTTPException(status_code=500, detail=str(e))