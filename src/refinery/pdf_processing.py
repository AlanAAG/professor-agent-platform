# src/refinery/pdf_processing.py

import fitz  # PyMuPDF
import os
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import logging # Use logging instead of print for better tracking

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv()

# --- Configure Gemini Client (for image descriptions) ---
image_model = None
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # Use the specific model capable of image input
    # Ensure billing is enabled for multi-modal models like 1.5-flash or pro-vision
    genai.configure(api_key=gemini_api_key)
    image_model = genai.GenerativeModel('gemini-1.5-flash') # Recommended model
    logging.info("Gemini multi-modal model configured for image descriptions.")
except Exception as e:
    logging.warning(f"Could not configure Gemini multi-modal model: {e}. Image descriptions will be unavailable.")

# --- Image Description Function ---
def generate_image_description(image_bytes: bytes) -> str:
    """Sends image bytes to a multi-modal LLM to get a description."""
    if not image_model:
        return "Image description unavailable (Model not configured)."

    logging.info("   Generating image description...")
    try:
        image_part = {
            "mime_type": "image/png", # Assuming PNG, PyMuPDF often extracts as PNG
            "data": image_bytes
        }
        prompt = "Describe this image in detail, focusing on elements relevant to an academic lecture, presentation slide, or document (e.g., charts, diagrams, key text, formulas, visual concepts)."

        # Add safety settings if needed, depending on content
        # safety_settings=[
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        #     # ... other categories
        # ]

        response = image_model.generate_content(
            [prompt, image_part],
            # safety_settings=safety_settings
            )
        description = response.text.strip()
        logging.info(f"      Generated description (length: {len(description)})")
        return description
    except Exception as e:
        logging.error(f"      Error generating image description: {e}")
        # Check if the error is due to safety settings
        # try:
        #     logging.error(f"      Response safety feedback: {response.prompt_feedback}")
        # except Exception:
        #     pass # No feedback available
        return "Error generating image description."

# --- Main PDF Processing Function ---
def process_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text, image descriptions, and links from a PDF, page by page.
    Returns a list of dictionaries, one per page, containing extracted data.
    """
    extracted_data_per_page = []
    doc = None # Initialize doc to None
    try:
        if not os.path.exists(pdf_path):
            logging.error(f"PDF file not found: {pdf_path}")
            return []
            
        doc = fitz.open(pdf_path)
        logging.info(f"-> Processing PDF: {os.path.basename(pdf_path)} ({len(doc)} pages)")

        for page_num in range(len(doc)):
            page_content = {"page_number": page_num + 1, "text": "", "images": [], "links": []}
            try:
                page = doc.load_page(page_num)

                # 1. Extract Text (using layout-aware extraction if possible)
                # Consider using page.get_text("blocks") or "dict" for more structure
                page_text = page.get_text("text", sort=True).strip() # Sort helps with reading order
                page_content["text"] = page_text if page_text else "" # Ensure empty string if no text

                # 2. Extract Images and Generate Descriptions
                image_list = page.get_images(full=True)
                if image_list:
                    logging.info(f"   Page {page_num + 1}: Found {len(image_list)} images.")
                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        if base_image and base_image.get("image"):
                            image_bytes = base_image["image"]
                            description = generate_image_description(image_bytes)
                            page_content["images"].append({
                                "index": img_index,
                                "description": description
                            })
                        else:
                            logging.warning(f"   Page {page_num + 1}: Could not extract image data for index {img_index}.")

                # 3. Extract Links (URIs)
                link_list = page.get_links()
                if link_list:
                    page_content["links"] = [link.get("uri", "") for link in link_list if link.get("kind") == fitz.LINK_URI and link.get("uri")]

                extracted_data_per_page.append(page_content)

            except Exception as page_err:
                logging.error(f"   Error processing page {page_num + 1} in {os.path.basename(pdf_path)}: {page_err}")
                # Add a placeholder or skip the page depending on desired robustness
                extracted_data_per_page.append({"page_number": page_num + 1, "text": "[Error processing page]", "images": [], "links": []})


        logging.info(f"✅ Finished extracting data from PDF: {os.path.basename(pdf_path)}")

    except Exception as e:
        logging.error(f"❌ Critical Error opening or processing PDF {pdf_path}: {e}")
        # Return whatever was processed before the critical error
        return extracted_data_per_page
        
    finally:
        if doc:
            doc.close() # Ensure the document is closed

    return extracted_data_per_page