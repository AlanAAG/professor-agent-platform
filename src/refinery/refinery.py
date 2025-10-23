# src/refinery/refinery.py

import os
import shutil
from dotenv import load_dotenv
import logging

# Import refinery sub-modules
from . import cleaning
from . import embedding
from . import pdf_processing # For handling PDFs

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv()

# --- Directory Configuration ---
# Define paths relative to the project root
RAW_TRANSCRIPT_DIR = "data/raw_transcripts/"
RAW_STATIC_DIR = "data/raw_static/"
RAW_PDF_DIR = "data/raw_pdfs/"
PROCESSED_DIR = "data/processed/" # Archive for successfully processed files
FAILED_DIR = "data/failed/"     # Archive for files that failed processing

# --- Ensure Output Directories Exist ---
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)


def move_file(source_path: str, destination_dir: str):
    """Safely moves a file, handling potential errors."""
    if not os.path.exists(source_path):
        logging.warning(f"Cannot move file - Source not found: {source_path}")
        return
    try:
        filename = os.path.basename(source_path)
        destination_path = os.path.join(destination_dir, filename)
        os.makedirs(destination_dir, exist_ok=True)
        shutil.move(source_path, destination_path)
        logging.info(f"Moved {filename} to {destination_dir}")
    except OSError as move_err:
        logging.error(f"Error moving {os.path.basename(source_path)} to {destination_dir}: {move_err}")

def run_refinery():
    """
    Main function for the Refinery pipeline.
    Processes raw transcripts, static text files, and PDFs,
    then cleans, chunks, and embeds them into the vector database.
    """
    logging.info("🧠 Refinery Pipeline Started...")

    # --- 1. Process Raw Transcripts ---
    logging.info("\n--- Processing Transcripts ---")
    if not os.path.exists(RAW_TRANSCRIPT_DIR):
        logging.warning(f"Directory not found: {RAW_TRANSCRIPT_DIR}. Skipping transcript processing.")
    else:
        for filename in os.listdir(RAW_TRANSCRIPT_DIR):
            if not filename.endswith(".txt"): continue
            filepath = os.path.join(RAW_TRANSCRIPT_DIR, filename)
            logging.info(f"-> Processing Transcript: {filename}")
            try:
                # Derive metadata from filename (e.g., "ClassName_YYYY-MM-DD_Title.txt")
                parts = filename.replace(".txt", "").split("_", 2) # Split max 3 times
                if len(parts) < 2:
                    raise ValueError("Unexpected transcript filename format.")
                class_name, date_str = parts[0], parts[1]
                # title = parts[2] if len(parts) > 2 else None # Optional title

                with open(filepath, "r", encoding='utf-8') as f:
                    raw_text = f.read()

                clean_text = cleaning.clean_transcript_with_llm(raw_text)
                if not clean_text: raise ValueError("Cleaning returned empty text.")

                metadata = {
                    "class_name": class_name,
                    "lecture_date": date_str,
                    "content_type": "lecture",
                    "source_file": filename
                }

                embedding.chunk_and_embed_text(clean_text, metadata)
                move_file(filepath, PROCESSED_DIR)

            except Exception as e:
                logging.error(f"❌ Failed to process transcript {filename}: {e}")
                move_file(filepath, FAILED_DIR)

    # --- 2. Process Raw Static Text Files (e.g., Syllabus) ---
    logging.info("\n--- Processing Static Text Files ---")
    if not os.path.exists(RAW_STATIC_DIR):
        logging.warning(f"Directory not found: {RAW_STATIC_DIR}. Skipping static file processing.")
    else:
        for filename in os.listdir(RAW_STATIC_DIR):
            if not filename.endswith(".txt"): continue
            filepath = os.path.join(RAW_STATIC_DIR, filename)
            logging.info(f"-> Processing Static File: {filename}")
            try:
                # Derive metadata (e.g., "ClassName_syllabus.txt")
                parts = filename.replace(".txt", "").split("_", 1)
                if len(parts) < 2: raise ValueError("Unexpected static filename format.")
                class_name, content_type_tag = parts[0], parts[1] # e.g., "syllabus"

                with open(filepath, "r", encoding='utf-8') as f:
                    static_text = f.read()

                if len(static_text) < 50:
                    logging.warning(f"   Skipping static file {filename} (minimal content).")
                    move_file(filepath, PROCESSED_DIR) # Move even if skipped embedding
                    continue

                metadata = {
                    "class_name": class_name,
                    "content_type": content_type_tag, # Use the tag from filename
                    "source_file": filename
                }

                embedding.chunk_and_embed_text(static_text, metadata)
                move_file(filepath, PROCESSED_DIR)

            except Exception as e:
                logging.error(f"❌ Failed to process static file {filename}: {e}")
                move_file(filepath, FAILED_DIR)

    # --- 3. Process Raw PDF Files ---
    logging.info("\n--- Processing PDFs ---")
    if not os.path.exists(RAW_PDF_DIR):
        logging.warning(f"Directory not found: {RAW_PDF_DIR}. Skipping PDF processing.")
    else:
        for filename in os.listdir(RAW_PDF_DIR):
            if not filename.lower().endswith(".pdf"): continue
            filepath = os.path.join(RAW_PDF_DIR, filename)
            logging.info(f"-> Processing PDF File: {filename}")
            try:
                # Derive class_name (adjust logic based on your naming convention)
                class_name = filename.split("_")[0] # Assumes "ClassName_..." format

                # Call the PDF processor module - gets list of page data dicts
                pages_data = pdf_processing.process_pdf(filepath)
                if not pages_data: raise ValueError("PDF processing returned no data.")

                logging.info(f"   Extracted data for {len(pages_data)} pages.")

                # Process data page by page
                for page_data in pages_data:
                    # --- Data Abstraction Step ---
                    combined_text = page_data.get("text", "")
                    image_descriptions = [
                        img.get("description", "")
                        for img in page_data.get("images", [])
                        if img.get("description") and "unavailable" not in img.get("description", "").lower() and "error" not in img.get("description", "").lower()
                    ] # Filter out failures
                    links = page_data.get("links", [])

                    # Append descriptions if they exist
                    if image_descriptions:
                        combined_text += "\n\n[Image Descriptions:]\n" + "\n".join(f"- {desc}" for desc in image_descriptions)

                    # --- Check Content Length ---
                    if len(combined_text) < 50:
                        logging.warning(f"   Skipping embedding for page {page_data.get('page_number', 'N/A')} (minimal content).")
                        continue

                    # --- Define Metadata ---
                    metadata = {
                        "class_name": class_name,
                        "content_type": "pdf_page", # Generic type, could refine from filename if needed
                        "source_file": filename,
                        "page_number": page_data.get("page_number"),
                        "links": links # Store links in metadata
                    }
                    metadata = {k: v for k, v in metadata.items() if v is not None} # Clean Nones

                    # --- Pass Abstracted Data to Embedding ---
                    logging.info(f"   Embedding content for page {metadata.get('page_number', 'N/A')}...")
                    embedding.chunk_and_embed_text(combined_text, metadata)

                # If loop completes without critical error for the PDF
                move_file(filepath, PROCESSED_DIR)

            except Exception as e:
                logging.error(f"❌ Failed to process PDF {filename}: {e}")
                move_file(filepath, FAILED_DIR)

    logging.info("\n🧠 Refinery Pipeline Finished.")


# --- Allow running this script directly for testing ---
if __name__ == "__main__":
    # Ensure necessary input folders exist for testing
    os.makedirs(RAW_TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(RAW_STATIC_DIR, exist_ok=True)
    os.makedirs(RAW_PDF_DIR, exist_ok=True)

    logging.info("Running full refinery pipeline locally...")
    run_refinery()
    logging.info("Local refinery run complete. Check logs and Supabase.")