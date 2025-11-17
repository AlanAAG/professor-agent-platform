# src/refinery/pdf_processing.py

import fitz  # PyMuPDF
import os
import base64
from dotenv import load_dotenv
import logging  # Use logging instead of print for better tracking
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import tempfile

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv()

# --- Configure Mistral Client via LangChain (for image descriptions) ---
image_model = None
vision_model_name = os.environ.get("MISTRAL_VISION_MODEL_NAME") or os.environ.get("MISTRAL_MODEL_NAME") or "pixtral-large-latest"
vision_temperature = float(os.environ.get("MISTRAL_VISION_TEMPERATURE", "0.2"))
try:
    mistral_api_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")

    image_model = ChatMistralAI(
        model=vision_model_name,
        api_key=mistral_api_key,
        temperature=vision_temperature,
    )
    logging.info(
        "Mistral vision model configured via LangChain for image descriptions (model=%s).",
        vision_model_name,
    )
except Exception as e:
    logging.warning(
        "Could not configure Mistral vision model via LangChain: %s. Image descriptions will be unavailable.",
        e,
    )

# --- Image Description Function ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
    reraise=True,
)
def generate_image_description(image_bytes: bytes) -> str:
    """Sends image bytes to a multi-modal LLM to get a description using LangChain."""
    if not image_model:
        return "Image description unavailable (Model not configured)."

    logging.info("   Generating image description...")
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

    try:
        response = image_model.invoke([message])
    except Exception as exc:
        logging.error("      Error generating image description: %s", exc, exc_info=True)
        raise

    # LangChain ChatModels typically return an AIMessage with .content
    description = getattr(response, "content", None)
    if not description:
        description = str(response)
    description = description.strip()
    logging.info(f"      Generated description (length: {len(description)})")
    return description

# --- Main PDF Processing Function ---


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _open_pdf_temp_copy(pdf_path: str) -> tuple[fitz.Document, str]:
    """Copy PDF to a NamedTemporaryFile and open via PyMuPDF; returns (doc, temp_path)."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    with open(pdf_path, "rb") as fsrc, tempfile.NamedTemporaryFile(prefix="pdf_proc_", suffix=".pdf", delete=False) as tmp:
        tmp.write(fsrc.read())
        temp_path = tmp.name
    try:
        doc = fitz.open(temp_path)
    except Exception:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        except Exception as cleanup_err:
            logging.warning(
                "Failed to remove temporary PDF copy at %s after open failure: %s",
                temp_path,
                cleanup_err,
                exc_info=True,
            )
        raise

    try:
        os.remove(temp_path)
    except FileNotFoundError:
        pass
    except Exception as cleanup_err:
        logging.warning(
            "Failed to remove temporary PDF copy at %s: %s",
            temp_path,
            cleanup_err,
            exc_info=True,
        )
    return doc, temp_path


def _fallback_pdfplumber_text(pdf_path: str) -> Optional[List[str]]:
    if not pdfplumber:
        logging.warning("pdfplumber not available; skipping fallback text extraction")
        return None
    try:
        texts: List[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = (page.extract_text() or "").strip()
                texts.append(t)
        return texts
    except Exception as e:  # pragma: no cover
        logging.warning(f"pdfplumber fallback failed: {e}")
        return None


def process_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Robust PDF extractor with retries, fallbacks, and partial success reporting.

    Returns list of dicts per page:
      {"page": 1, "text": "...", "status": "success" | "ocr_success" | "ocr_failed" | "error", "error": "..."}
    Additional legacy keys maintained for compatibility: page_number, images, links
    """
    results: List[Dict[str, Any]] = []
    temp_path: Optional[str] = None
    doc: Optional[fitz.Document] = None

    try:
        doc, temp_path = _open_pdf_temp_copy(pdf_path)
        if temp_path and not os.path.exists(temp_path):
            temp_path = None
        logging.info(f"-> Processing PDF: {os.path.basename(pdf_path)} ({len(doc)} pages)")

        for page_idx in range(len(doc)):
            page_info: Dict[str, Any] = {
                "page": page_idx + 1,
                "page_number": page_idx + 1,  # legacy
                "text": "",
                "status": "",
                "images": [],  # legacy
                "links": [],   # legacy
            }
            try:
                page = doc.load_page(page_idx)

                # Text extraction (layout-aware)
                text = (page.get_text("text", sort=True) or "").strip()
                # Links (legacy)
                try:
                    for link in page.get_links() or []:
                        if link.get("kind") == fitz.LINK_URI and link.get("uri"):
                            page_info["links"].append(link.get("uri"))
                except Exception:
                    pass

                # Images -> Vision description as a last resort OCR-like fallback
                images = page.get_images(full=True)
                if images:
                    logging.info(f"   Page {page_idx + 1}: Found {len(images)} images.")
                for img_index, img_info in enumerate(images or []):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        if base_image and base_image.get("image"):
                            image_bytes = base_image["image"]
                            description = generate_image_description(image_bytes)
                            page_info["images"].append({"index": img_index, "description": description})
                    except Exception as image_err:
                        logging.warning(
                            "      Skipping image %s on page %s due to error: %s",
                            img_index,
                            page_idx + 1,
                            image_err,
                            exc_info=True,
                        )
                        continue

                if text:
                    page_info["text"] = text
                    page_info["status"] = "success"
                    results.append(page_info)
                    continue

                # Fallback 1: pdfplumber textual extraction
                fallback_texts = _fallback_pdfplumber_text(pdf_path)
                if fallback_texts is not None:
                    ft = (fallback_texts[page_idx] if page_idx < len(fallback_texts) else "").strip()
                    if ft:
                        page_info["text"] = ft
                        page_info["status"] = "success"
                        results.append(page_info)
                        continue

                # Fallback 2: image render + vision model for OCR-like content
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    description = generate_image_description(img_bytes)
                    if description and "unavailable" not in description.lower() and "error" not in description.lower():
                        page_info["text"] = description
                        page_info["status"] = "ocr_success"
                    else:
                        page_info["status"] = "ocr_failed"
                        page_info["error"] = "Vision OCR returned no usable text"
                except Exception as e:
                    page_info["status"] = "ocr_failed"
                    page_info["error"] = f"Vision OCR failed: {e}"
                results.append(page_info)

            except Exception as page_err:
                page_info["text"] = ""
                page_info["status"] = "error"
                page_info["error"] = str(page_err)
                results.append(page_info)

        logging.info(f"? Finished extracting data from PDF: {os.path.basename(pdf_path)}")

    except Exception as e:
        logging.error(f"? Critical error opening/processing PDF {pdf_path}: {e}")
        # Return partial results if any
        return results
    finally:
        if doc:
            try:
                doc.close()
            except Exception as close_err:
                logging.warning("Failed to close PDF document: %s", close_err, exc_info=True)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            except Exception as cleanup_err:
                logging.warning("Failed to remove temporary PDF copy at %s: %s", temp_path, cleanup_err, exc_info=True)

    return results