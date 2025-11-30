# src/refinery/cleaning.py

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Ensure environment is loaded
load_dotenv()

# Define the cleaning prompt
CLEANING_PROMPT = """You are an expert transcript editor. Your task is to clean up the following lecture transcript.
Remove filler words (um, uh, like, you know), fix grammatical errors, and ensure the text flows logically while preserving the original meaning and technical terms.
Do not summarize. Return the full cleaned text.

Transcript:
{text}
"""

# Initialize the model globally (lazy loading pattern)
model = None

try:
    # Support both standard key names
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        # Don't crash immediately, just log error. Validation happens on use.
        logging.warning("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in environment.")
    else:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", # or "gemini-1.5-flash"
            google_api_key=api_key,
            temperature=0.45,
            timeout=300,        # <--- Critical for large files (5 mins)
            max_retries=3       # <--- Retry on failure
        )
        logging.info("Gemini cleaning model initialized successfully.")
except Exception as e:
    logging.error(f"Error configuring Gemini via LangChain: {e}")
    model = None

def clean_transcript_with_llm(raw_text: str) -> str:
    """
    Cleans a raw transcript using the configured LLM.
    """
    if not raw_text:
        return ""
    
    if not model:
        logging.warning("LLM unavailable or failed to initialize. Falling back to local cleaning (basic whitespace).")
        return " ".join(raw_text.split())

    try:
        prompt = ChatPromptTemplate.from_template(CLEANING_PROMPT)
        chain = prompt | model
        
        logging.info(f"-> Sending {len(raw_text)} characters to Gemini for cleaning...")
        result = chain.invoke({"text": raw_text})
        
        cleaned_text = result.content
        logging.info("-> Transcript cleaned successfully.")
        return cleaned_text

    except Exception as e:
        logging.error(f"LLM cleaning failed: {e}")
        # Fallback to basic cleaning
        return " ".join(raw_text.split())