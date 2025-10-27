from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import cohere
from src.shared.utils import EMBEDDING_MODEL_NAME, cohere_rerank, retrieve_rag_documents
import os
import json
from typing import List, Dict, Optional

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Configure re-ranking (Cohere) and embedding model name for consistency
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

# IMPORTANT: Keep this aligned with src/refinery/embedding.py
EMBEDDING_MODEL = EMBEDDING_MODEL_NAME

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    selectedClass: Optional[str] = None
    persona: str = "balanced"

class RAGRequest(BaseModel):
    query: str
    selectedClass: Optional[str] = None

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "AI Tutor API"}

@app.post("/api/rag-search")
async def rag_search(request: RAGRequest):
    try:
        # Standardized retrieval via shared utility (Supabase RPC + Gemini embeddings)
        documents = retrieve_rag_documents(
            query=request.query,
            selected_class=request.selectedClass,
            match_count=5,
            match_threshold=0.7,
        )
        # Optional re-ranking (no-op if Cohere not configured)
        documents = cohere_rerank(request.query, documents)
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    try:
        # Get RAG context
        user_messages = [m for m in request.messages if m.get("role") == "user"]
        last_query = user_messages[-1]["content"] if user_messages else ""
        
        rag_response = await rag_search(RAGRequest(
            query=last_query,
            selectedClass=request.selectedClass
        ))
        documents = rag_response["documents"]
        
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
        
        system_prompt = f"""{personas.get(request.persona, personas['balanced'])}

COURSE MATERIALS:
{context}

RULES:
- ONLY use information from the provided course materials
- If information is not in the materials, say so
- Cite sources when referencing specific content
- Be conversational but accurate"""
        
        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        for msg in request.messages:
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
