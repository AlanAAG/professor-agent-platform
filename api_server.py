from fastapi import FastAPI, HTTPException, Depends, Security, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from src.shared.utils import EMBEDDING_MODEL_NAME, cohere_rerank, retrieve_rag_documents
import google.generativeai as genai
import os
import json
from typing import List, Dict, Optional
from datetime import datetime

# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS Setup (Robust, Environment-based) ---
# Read allowed origins from environment variable `ALLOWED_ORIGINS`.
# Supports comma-separated string.
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    # Allow Content-Type, Authorization, and the custom X-API-Key header
    allow_headers=["x-api-key", "content-type", "authorization"],
)

# --- Initialize clients ---
# Uses GEMINI_API_KEY environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Optional Supabase client for health checks
try:
    from src.shared.utils import _get_supabase_client
except Exception:
    _get_supabase_client = None  # type: ignore

supabase = None
if _get_supabase_client is not None:
    try:
        supabase = _get_supabase_client()
    except Exception:
        supabase = None

# IMPORTANT: Keep this aligned with src/refinery/embedding.py
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME

# --- API Key Authentication Setup ---
api_key_header = APIKeyHeader(name="x-api-key", auto_error=True)

def get_api_key(api_key: str = Security(api_key_header)) -> str:
    expected = os.getenv("SECRET_API_KEY")
    if not expected:
        # Service misconfiguration; reject until configured
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key != expected:
        # Use 403 Forbidden since the key is likely just wrong
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

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
    return {
        "status": "healthy",
        "service": "AI Tutor API",
        "version": "1.0.0-beta",
        "timestamp": datetime.now().isoformat(),
        "supabase_connected": supabase is not None,
        "model_loaded": model is not None,
    }

@app.get("/health")
async def health():
    """Dedicated health endpoint for uptime monitoring"""
    return {"status": "ok"}

@app.post("/api/rag-search")
async def rag_search(payload: RAGRequest, api_key: str = Depends(get_api_key)):
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
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_stream(request: Request, payload: ChatRequest, api_key: str = Depends(get_api_key)):
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
            
            # Stream AI response
            response = model.generate_content(
                [m["content"] for m in messages],
                stream=True
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
        raise HTTPException(status_code=500, detail=str(e))