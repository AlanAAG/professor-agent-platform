# src/refinery/embedding.py

import os
from supabase.client import Client, create_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Ensures .env file is read when running locally.
load_dotenv()

# --- 1. Initialize Clients ---
supabase = None
embeddings_model = None
vector_store = None # Initialize vector_store as None initially

try:
    # --- Check and Load Environment Variables ---
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY must be set in .env for embeddings")

    # --- Initialize Supabase Client ---
    supabase = create_client(supabase_url, supabase_key)
    print("Supabase client initialized.")

    # --- Initialize Embedding Model ---
    # Make sure you have the correct model name for embeddings.
    # 'models/embedding-001' is the standard Gemini embedding model.
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key
    )
    print("Gemini embedding model initialized.")

    # --- Initialize the Vector Store Client ---
    # Moved initialization here after ensuring supabase and embeddings_model are valid
    vector_store = SupabaseVectorStore(
        client=supabase,
        table_name="documents", # The table you created in Supabase SQL Editor
        embedding=embeddings_model
    )
    print("SupabaseVectorStore initialized.")

    print("✅ Embedding clients configured successfully.")

except Exception as e:
    # --- Improved Error Handling ---
    print(f"❌ FATAL ERROR: Could not initialize embedding clients.")
    print(f"   Error details: {e}")
    print("   Please double-check your .env file for correct SUPABASE_URL, SUPABASE_KEY, and GEMINI_API_KEY.")
    # Exit the script immediately if initialization fails.
    exit(1)

# --- 3. Initialize the Text Splitter ---
# This object is responsible for "chunking" your large text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, # Max characters per chunk (adjust if needed)
    chunk_overlap=200, # Characters to overlap between chunks (helps maintain context)
    length_function=len
)
print("Text splitter initialized.")


# --- Main Function to be Called by Other Scripts ---
def chunk_and_embed_text(clean_text: str, metadata: dict):
    """
    Takes clean text, splits it into chunks, and embeds it
    into the Supabase vector store with metadata.
    """
    if not vector_store:
        # This check is mostly redundant now due to exit(1) above, but good safety measure.
        raise EnvironmentError("Vector store is not initialized. Exiting.")
    if not clean_text:
        print("-> Skipping embedding: No clean text provided.")
        return

    print(f"-> Chunking and embedding for: {metadata.get('class_name')}")

    # 1. Split the clean text into LangChain "Document" objects (chunks)
    documents = text_splitter.create_documents([clean_text])
    print(f"   Split text into {len(documents)} chunks.")

    # 2. Add your crucial metadata dictionary to *every single chunk*
    for doc in documents:
        doc.metadata = metadata

    # 3. Add all prepared documents to Supabase
    # LangChain's SupabaseVectorStore handles calculating embeddings
    # and saving everything to your 'documents' table.
    try:
        vector_store.add_documents(documents)
        print(f"✅ Embedding successful for {len(documents)} chunks.")
    except Exception as e:
        print(f"❌ Error during Supabase add_documents: {e}")
        # Depending on the error, you might want to raise it, retry, or log it.
        # Re-raising ensures the calling script knows about the failure.
        raise


def check_if_embedded(filter: dict) -> bool:
    """
    (Advanced) Checks if a document (like a syllabus) with specific metadata
    already exists in the database.
    """
    # For now, just return False so it always embeds new static files.
    # You can implement this later using vector_store.similarity_search()
    # or direct Supabase queries if needed.
    return False

# --- Optional: Test Initialization ---
# You can run this file directly (`python src/refinery/embedding.py`)
# to check if the clients initialize correctly without errors.
if __name__ == "__main__":
    print("\n--- Running local initialization test for embedding.py ---")
    if supabase and embeddings_model and vector_store:
        print("✅ Initialization appears successful.")
    else:
        # The exit(1) in the try block should prevent this, but just in case.
        print("❌ Initialization failed. Check error messages above.")