# src/refinery/pdf_processing.py

import fitz  # PyMuPDF
import os
import base64
from dotenv import load_dotenv
import logging  # Use logging instead of print for better tracking
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv()

# --- Configure Gemini Client via LangChain (for image descriptions) ---
image_model = None
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    # Use a multi-modal capable Gemini model
    image_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
    )
    logging.info("Gemini multi-modal model configured via LangChain for image descriptions.")
except Exception as e:
    logging.warning(
        f"Could not configure Gemini multi-modal model via LangChain: {e}. Image descriptions will be unavailable."
    )

# --- Image Description Function ---
def generate_image_description(image_bytes: bytes) -> str:
    """Sends image bytes to a multi-modal LLM to get a description using LangChain."""
    if not image_model:
        return "Image description unavailable (Model not configured)."

    logging.info("   Generating image description...")
    try:
        # Encode image as data URL for LangChain multi-modal message
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            "Describe this image in detail, focusing on elements relevant to an academic "
            "lecture, presentation slide, or document (e.g., charts, diagrams, key text, "
            "formulas, visual concepts)."
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
            ]
        )

        response = image_model.invoke([message])
        # LangChain ChatModels typically return an AIMessage with .content
        description = getattr(response, "content", None)
        if not description:
            description = str(response)
        description = description.strip()
        logging.info(f"      Generated description (length: {len(description)})")
        return description
    except Exception as e:
        logging.error(f"      Error generating image description: {e}")
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