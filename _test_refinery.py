import os
from dotenv import load_dotenv
# Use relative imports that work when run from project root: `python _test_refinery.py`
from src.refinery import cleaning, embedding  

# 1. Load all .env variables
load_dotenv()

def run_test():
    """Runs a local test of the cleaning and embedding refinery pipeline."""
    
    # 2. Define your test file and metadata
    TEST_FILE_PATH = "data/raw_transcripts/test.txt"
    TEST_METADATA = {
        "class_name": "Statistics",
        "lecture_date": "2025-10-17",
        "content_type": "lecture"
    }
    
    # --- Ensure required data directories exist ---
    if not os.path.exists("data/raw_transcripts"):
        os.makedirs("data/raw_transcripts", exist_ok=True)
    
    # --- Create a placeholder test file if it doesn't exist yet ---
    if not os.path.exists(TEST_FILE_PATH):
        # Placeholder content (must be written before attempting to read)
        with open(TEST_FILE_PATH, "w") as f:
            f.write("0:01 This is a test transcript with timestamps to check the cleaning. 1:45 The goal is to see if the cleaning and embedding functions work.")
        print(f"Created placeholder file at {TEST_FILE_PATH}")


    # 3. Read the raw text
    print(f"Loading raw file: {TEST_FILE_PATH}")
    with open(TEST_FILE_PATH, "r") as f:
        raw_text = f.read()

    # --- STEP 1: CLEAN ---
    print("\n--- Starting Cleaning ---")
    try:
        clean_text = cleaning.clean_transcript_with_llm(raw_text)
    except Exception as e:
        print(f"❌ Cleaning failed: {e}. Check MISTRAL_API_KEY.")
        return
    
    if not clean_text:
        print("Cleaning returned empty content. Aborting.")
        return

    print("--- CLEANING COMPLETE ---")

    # --- STEP 2: EMBED ---
    print("\n--- Starting Embedding ---")
    try:
        # Note: This will call the Mistral Embeddings API and write to Supabase.
        embedding.chunk_and_embed_text(clean_text, TEST_METADATA)
        
        print("\n🎉 ----- REFINERY TEST COMPLETE ----- 🎉")
        print("Check your 'documents' table in Supabase to see the chunks!")
    
    except EnvironmentError as e:
        print(f"\n❌ --- EMBEDDING FAILED (Environment) --- ❌")
        print(f"Check error: {e}")
    except Exception as e:
        print(f"\n❌ --- EMBEDDING FAILED (Runtime) --- ❌")
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_test()