# src/refinery/embedding.py

import os
import logging
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
    logging.info("Supabase client initialized.")

    # --- Initialize Embedding Model ---
    # Make sure you have the correct model name for embeddings.
    # 'models/embedding-001' is the standard Gemini embedding model.
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=gemini_api_key
    )
    logging.info("Gemini embedding model initialized.")

    # --- Initialize the Vector Store Client ---
    # Moved initialization here after ensuring supabase and embeddings_model are valid
    vector_store = SupabaseVectorStore(
        client=supabase,
        table_name="documents", # The table you created in Supabase SQL Editor
        embedding=embeddings_model
    )
    logging.info("SupabaseVectorStore initialized.")

    logging.info("✅ Embedding clients configured successfully.")

except Exception as e:
    # --- Improved Error Handling ---
    logging.error("Could not initialize embedding clients.")
    logging.error(f"Error details: {e}")
    logging.error("Please double-check your .env file for correct SUPABASE_URL, SUPABASE_KEY, and GEMINI_API_KEY.")
    # Do not exit here; allow importing modules that handle missing clients gracefully.

# --- 3. Initialize the Text Splitter ---
# This object is responsible for "chunking" your large text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, # Max characters per chunk (adjust if needed)
    chunk_overlap=200, # Characters to overlap between chunks (helps maintain context)
    length_function=len
)
logging.info("Text splitter initialized.")


# --- Main Function to be Called by Other Scripts ---
def chunk_and_embed_text(clean_text: str, metadata: dict):
    """
    Takes clean text, splits it into chunks, and embeds it
    into the Supabase vector store with metadata.
    """
    if not vector_store:
        # This check is important because initialization may have failed gracefully.
        raise EnvironmentError("Vector store is not initialized.")
    if not clean_text:
        logging.info("-> Skipping embedding: No clean text provided.")
        return

    logging.info(f"-> Chunking and embedding for: {metadata.get('class_name')}")

    # 1. Split the clean text into LangChain "Document" objects (chunks)
    documents = text_splitter.create_documents([clean_text])
    logging.info(f"   Split text into {len(documents)} chunks.")

    # 2. Add your crucial metadata dictionary to *every single chunk*
    for doc in documents:
        doc.metadata = metadata

    # 3. Add all prepared documents to Supabase
    # LangChain's SupabaseVectorStore handles calculating embeddings
    # and saving everything to your 'documents' table.
    try:
        vector_store.add_documents(documents)
        logging.info(f"✅ Embedding successful for {len(documents)} chunks.")
    except Exception as e:
        logging.error(f"❌ Error during Supabase add_documents: {e}")
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


def check_if_embedded_recently(filter: dict, max_age_days: int = 90) -> bool:
    """
    Placeholder implementation to determine if content matching the filter
    has been embedded recently. Returns False to force embedding until a
    proper Supabase query is implemented.

    Expected filter keys include metadata such as 'class_name' and 'content_type'.
    """
    # Future improvement: query Supabase 'documents' table using the client
    # for metadata filters and a created_at/updated_at timestamp comparison.
    return False

# --- Optional: Test Initialization ---
# You can run this file directly (`python src/refinery/embedding.py`)
# to check if the clients initialize correctly without errors.
if __name__ == "__main__":
    logging.info("\n--- Running local initialization test for embedding.py ---")
    if supabase and embeddings_model and vector_store:
        logging.info("✅ Initialization appears successful.")
    else:
        logging.error("❌ Initialization failed. Check error messages above.")