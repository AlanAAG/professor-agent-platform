import os
import json
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import existing RAG logic
from src.app import rag_core

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
app = FastAPI(title="Professor AI API")

PERSONA_FILE_PATH = os.path.join(os.path.dirname(__file__), "app/persona.json")
PERSONAS: dict = {}

# --- CORS Middleware ---
# Allow all origins for now; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models for Request/Response ---
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    class_name: str
    chat_history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    # Future: include sources if rag_core returns them


# --- Load Personas on Startup ---
@app.on_event("startup")
def load_personas() -> None:
    global PERSONAS
    logging.info("Loading personas...")
    try:
        with open(PERSONA_FILE_PATH, "r", encoding="utf-8") as f:
            PERSONAS = json.load(f)
        logging.info("Successfully loaded %d personas.", len(PERSONAS))
    except Exception as e:
        logging.critical("Failed to load persona.json: %s", e)
        # Keep PERSONAS empty; /chat will return 404 for missing class


# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint that receives a question and returns an AI-generated answer."""
    logging.info("Received chat request for class: %s", request.class_name)

    # 1. Get persona for the requested class
    selected_persona = PERSONAS.get(request.class_name)
    if not selected_persona:
        logging.warning("Persona not found for class: %s", request.class_name)
        raise HTTPException(status_code=404, detail=f"Persona for class '{request.class_name}' not found.")

    # 2. Convert chat history to the format rag_core expects (list[dict])
    history_list_of_dicts = [msg.model_dump() for msg in request.chat_history]

    try:
        # 3. Call existing RAG logic
        answer = rag_core.get_rag_response(
            question=request.question,
            class_name=request.class_name,
            persona=selected_persona,
            chat_history=history_list_of_dicts,
        )
        return ChatResponse(answer=answer)
    except HTTPException:
        # Re-raise FastAPI exceptions untouched
        raise
    except Exception as e:
        logging.error("Error in RAG core: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing your request: {e}")


# --- Health Check Endpoint ---
@app.get("/")
def read_root() -> dict:
    return {"status": "Professor AI API is running"}
