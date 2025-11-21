# in src/refinery/cleaning.py
import os
import logging
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.shared.provider_config import (
    get_chat_model_name,
    get_llm_provider,
    get_mistral_api_key,
    get_mistral_base_url,
    is_mistral,
)

# --- Load Environment Variables ---
load_dotenv() 

# --- Configure the LLM Client (using LangChain) ---
mistral_model_name = os.environ.get("MISTRAL_MODEL_NAME", "mistral-large-latest")
try:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")

    model = ChatMistralAI(
        model=mistral_model_name,
        api_key=api_key,
        temperature=0.45,
    )
    logging.info("Mistral model configured successfully (via LangChain).")
except Exception as e:
    logging.error(f"Error configuring Mistral via LangChain: {e}")
    model = None

def _clean_transcript_locally(raw_text: str) -> str:
    """
    Heuristic, offline cleaner for transcripts when no LLM is available.
    - Removes standalone timestamps like "0:01" or "1:03:45" on their own lines
    - Removes inline timestamps of the form HH:MM(:SS)? when surrounded by whitespace
    - Normalizes whitespace and creates readable paragraphs
    - Adds terminal punctuation where obviously missing
    """
    if not raw_text:
        return ""

    import re

    text = raw_text

    # Remove standalone timestamp lines
    text = re.sub(r"(?m)^\s*\d{1,2}:\d{2}(?::\d{2})?\s*$", "", text)

    # Remove inline timestamps that appear as isolated tokens
    text = re.sub(r"(?<!\d)(\b\d{1,2}:\d{2}(?::\d{2})?\b)(?!\d)", " ", text)

    # Collapse excessive whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # drop empties

    # Join short lines into paragraphs
    paragraphs: list[str] = []
    current: list[str] = []
    for ln in lines:
        if len(ln.split()) <= 2:  # likely leftover fragment, treat as break
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(ln)
    if current:
        paragraphs.append(" ".join(current))

    # Add terminal punctuation if clearly missing
    def ensure_punct(p: str) -> str:
        if not p:
            return p
        if p[-1] in ".!?":
            return p
        return p + "."

    paragraphs = [ensure_punct(p) for p in paragraphs]

    # Basic capitalization for sentence starts (conservative)
    def cap_sentence_start(p: str) -> str:
        return re.sub(r"(^|[\.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), p)

    paragraphs = [cap_sentence_start(p) for p in paragraphs]

    return "\n\n".join(paragraphs).strip()

def _get_cleaning_prompt() -> ChatPromptTemplate:
    """
    Creates the system prompt template for the LLM to clean a transcript.
    """
    prompt_template = """
    You are an expert transcript editor. Your task is to take the following raw, messy 
    lecture transcript and reformat it into a clean, punctuated, and highly 
    readable document.

    Follow these rules precisely:
    1.  **Remove all timestamps:** Any text pattern like "0:01", "1:03:45", or "59:07" 
        must be completely removed.
    2.  **Fix Punctuation and Capitalization:** Add sentence-ending punctuation (periods, 
        question marks) and fix capitalization (e.g., "i" to "I", start of new sentences).
    3.  **Create Paragraphs:** The input may be one giant block of text. Your output 
        MUST be structured into logical paragraphs. Break lines where a new idea or 
        speaker begins.
    4.  **Correct Obvious Spelling:** Correct clear spelling mistakes (e.g., "teh" to "the").
    5.  **Preserve Core Content:** Do NOT change technical terms, numbers, or the 
        speaker's core meaning, even if it seems like a mistake. Preserve the 
        original information accurately.

    The final output should be *only* the cleaned transcript. Do not add any 
    commentary, pre-amble, or post-amble like "Here is the cleaned transcript:".

    --- RAW TRANSCRIPT ---
    {raw_text}
    ---

    CLEANED TRANSCRIPT:
    """
    return ChatPromptTemplate.from_template(prompt_template)

def clean_transcript_with_llm(raw_text: str) -> str:
    """
    Sends the raw transcript to an LLM for cleaning using LangChain.
    """
    if not raw_text or len(raw_text) < 50:
        logging.info("-> Skipping cleaning: No significant text found.")
        return ""

    logging.info(f"-> Sending {len(raw_text)} characters to LLM for cleaning (via LangChain)...")
    
    try:
        if not model:
            raise RuntimeError("LLM not configured")

        # Create the prompt and the "chain"
        prompt = _get_cleaning_prompt()
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser

        # Invoke the chain with the raw text
        clean_text = chain.invoke({"raw_text": raw_text})

        return clean_text.strip()
        
    except Exception as e:
        logging.warning(f"LLM unavailable or failed ({e}). Falling back to local cleaning.")
        return _clean_transcript_locally(raw_text)

# --- Local Test ---
# (This test block remains the same)
if __name__ == "__main__":
    logging.info("Running local test for cleaning.py...")
    
    TEST_FILE_PATH = "data/raw_transcripts/test.txt" 
    
    if not os.path.exists(TEST_FILE_PATH):
        logging.error(f"Test file not found at {TEST_FILE_PATH}")
    else:
        with open(TEST_FILE_PATH, "r") as f:
            sample_text = f.read()
            
        clean_text = clean_transcript_with_llm(sample_text)
        
        logging.info("\n--- CLEANING COMPLETE ---")
        print(clean_text)