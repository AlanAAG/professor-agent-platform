# in src/refinery/cleaning.py
import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Load Environment Variables ---
load_dotenv() 

# --- Configure the LLM Client (using LangChain) ---
try:
    # We use the LangChain wrapper now
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    logging.info("Gemini model configured successfully (via LangChain).")
except Exception as e:
    logging.error(f"Error configuring Gemini via LangChain: {e}")
    model = None

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
    if not model:
        raise EnvironmentError("Gemini model is not configured. Check API key.")

    if not raw_text or len(raw_text) < 50:
        logging.info("-> Skipping cleaning: No significant text found.")
        return ""

    logging.info(f"-> Sending {len(raw_text)} characters to LLM for cleaning (via LangChain)...")
    
    try:
        # Create the prompt and the "chain"
        prompt = _get_cleaning_prompt()
        output_parser = StrOutputParser()
        chain = prompt | model | output_parser
        
        # Invoke the chain with the raw text
        clean_text = chain.invoke({"raw_text": raw_text})
        
        return clean_text.strip()
        
    except Exception as e:
        logging.error(f"❌ LLM Cleaning Error: {e}")
        return ""

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