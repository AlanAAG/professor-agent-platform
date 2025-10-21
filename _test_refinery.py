import os
from dotenv import load_dotenv
from src.refinery import cleaning, embedding  # <-- Import both

# 1. Load all .env variables
load_dotenv()

def run_test():
    # 2. Define your test file and metadata
    TEST_FILE_PATH = "data/raw_transcripts/test.txt"
    TEST_METADATA = {
        "class_name": "Statistics",
        "lecture_date": "2025-10-17",
        "content_type": "lecture"
    }

    # 3. Read the raw text
    print(f"Loading raw file: {TEST_FILE_PATH}")
    with open(TEST_FILE_PATH, "r") as f:
        raw_text = f.read()

    # --- STEP 1: CLEAN (Same as before) ---
    clean_text = cleaning.clean_transcript_with_llm(raw_text)
    
    if not clean_text:
        print("Cleaning failed. Aborting.")
        return

    print("--- CLEANING COMPLETE ---")

    # --- STEP 2: EMBED (New Step) ---
    try:
        embedding.chunk_and_embed_text(clean_text, TEST_METADATA)
        
        print("\n🎉 ----- REFINERY TEST COMPLETE ----- 🎉")
        print("Check your 'documents' table in Supabase to see the chunks!")
    
    except Exception as e:
        print(f"\n❌ --- EMBEDDING FAILED --- ❌")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_test()