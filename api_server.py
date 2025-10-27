"""
FastAPI server for TETR AI Tutor
Exposes API endpoints for chat and RAG search
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import json
import asyncio
from google.generativeai import GenerativeModel, configure
from supabase import create_client, Client
import cohere

# Initialize FastAPI
app = FastAPI(title="TETR AI Tutor API")

# Add CORS middleware to allow requests from Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EXTERNAL_SUPABASE_URL = os.getenv("EXTERNAL_SUPABASE_URL")
EXTERNAL_SUPABASE_SERVICE_KEY = os.getenv("EXTERNAL_SUPABASE_SERVICE_KEY")

# Initialize clients
if GOOGLE_API_KEY:
    configure(api_key=GOOGLE_API_KEY)
    
if COHERE_API_KEY:
    cohere_client = cohere.Client(COHERE_API_KEY)
    
if EXTERNAL_SUPABASE_URL and EXTERNAL_SUPABASE_SERVICE_KEY:
    supabase: Client = create_client(EXTERNAL_SUPABASE_URL, EXTERNAL_SUPABASE_SERVICE_KEY)

# Request/Response Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    selectedClass: str
    persona: str

class RAGSearchRequest(BaseModel):
    query: str
    selectedClass: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "TETR AI Tutor API",
        "endpoints": {
            "chat": "/api/chat",
            "rag_search": "/api/rag-search"
        }
    }

# RAG Search endpoint
@app.post("/api/rag-search")
async def rag_search(request: RAGSearchRequest):
    """
    Search for relevant course materials using vector similarity
    """
    try:
        if not EXTERNAL_SUPABASE_URL or not EXTERNAL_SUPABASE_SERVICE_KEY:
            raise HTTPException(status_code=500, detail="Supabase not configured")
        
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="Google API key not configured")
        
        # Generate embedding using Gemini
        model = GenerativeModel('models/embedding-001')
        embedding_result = model.embed_content(
            content=request.query,
            task_type="retrieval_query"
        )
        embedding = embedding_result['embedding']
        
        # Search Supabase vector database
        response = supabase.rpc(
            'match_documents',
            {
                'query_embedding': embedding,
                'match_threshold': 0.7,
                'match_count': 5,
                'filter_class': request.selectedClass
            }
        ).execute()
        
        documents = response.data if response.data else []
        
        # Re-rank with Cohere if available
        if COHERE_API_KEY and documents:
            doc_texts = [doc['content'] for doc in documents]
            rerank_response = cohere_client.rerank(
                model='rerank-english-v3.0',
                query=request.query,
                documents=doc_texts,
                top_n=5
            )
            
            # Reorder documents based on Cohere scores
            reranked_docs = []
            for result in rerank_response.results:
                doc = documents[result.index]
                doc['similarity'] = result.relevance_score
                reranked_docs.append(doc)
            documents = reranked_docs
        
        print(f"Found {len(documents)} documents for query: {request.query[:50]}...")
        
        return {"documents": documents}
        
    except Exception as e:
        print(f"RAG search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Stream AI responses using Gemini with RAG context
    """
    try:
        if not GOOGLE_API_KEY:
            raise HTTPException(status_code=500, detail="Google API key not configured")
        
        # Get course context using RAG search
        last_user_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
        
        course_context = ""
        sources = []
        
        if last_user_message:
            # Call our own RAG endpoint
            rag_result = await rag_search(RAGSearchRequest(
                query=last_user_message.content,
                selectedClass=request.selectedClass
            ))
            
            if rag_result["documents"]:
                sources = rag_result["documents"]
                course_context = "\n\nRelevant course materials:\n" + "\n\n".join([
                    f"[{idx + 1}] {doc.get('metadata', {}).get('title', 'Document')} ({doc.get('metadata', {}).get('class_name', '')})\n{doc['content']}"
                    for idx, doc in enumerate(sources)
                ])
        
        # Build system prompt
        system_prompt = f"""You are an AI tutor for {request.selectedClass}. {request.persona}

CRITICAL INSTRUCTIONS:
- You MUST ONLY use information from the provided course materials below
- DO NOT use any external knowledge or information from your training
- If the course materials don't contain the answer, say "I don't have information about that in the course materials"
- When answering, ALWAYS reference the source material by number [1], [2], etc.
- Stay strictly within the scope of the course topic

FORMATTING RULES (MANDATORY):
- NEVER use markdown bold formatting (**text**) - just write plain text
- Keep answers SHORT and DIGESTIBLE - students don't like long walls of text
- Use bullet points and short paragraphs (2-3 sentences max)
- Break complex ideas into simple, bite-sized pieces
- NEVER change the course name "{request.selectedClass}" - always use it exactly as provided

Your role:
- Answer questions using ONLY the course materials provided
- Keep responses concise and student-friendly
- Break down complex topics into easy chunks
- Help students learn efficiently without overwhelming them
{course_context if course_context else "\n\nNo course materials were found for this query. Inform the student that you need course content to answer their question."}

Keep responses SHORT, clear, and conversational. Students prefer quick, actionable answers over long explanations."""
        
        # Initialize Gemini model
        model = GenerativeModel('gemini-2.5-flash')
        
        # Convert messages to Gemini format
        chat_history = []
        for msg in request.messages[:-1]:  # Exclude last message
            chat_history.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [msg.content]
            })
        
        # Start chat with history
        chat = model.start_chat(history=chat_history)
        
        # Stream response
        async def generate_stream():
            # Send sources first if available
            if sources:
                sources_data = json.dumps({"sources": sources})
                yield f"data: {json.dumps({'choices': [{'delta': {'content': '', 'sources': sources_data}}]})}\n\n"
            
            # Stream AI response
            response = chat.send_message(
                f"{system_prompt}\n\nUser: {last_user_message.content}",
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk.text}}]})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
