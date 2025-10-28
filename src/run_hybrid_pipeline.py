# src/run_hybrid_pipeline.py

import os
import datetime
import logging
import tempfile
from dotenv import load_dotenv
from selenium import webdriver

# --- Import project modules (package-qualified for -m execution) ---
from src.harvester import navigation, scraping, config
from src.refinery import cleaning, embedding, pdf_processing  # Added pdf_processing
from selenium.webdriver.common.by import By
from src.shared import utils

# --- Setup Logging ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pipeline.log"), # Log to a file
        logging.StreamHandler() # Also print to console
    ]
)

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
# Directories for saving raw files locally (primarily for testing/backup)
# In the stateless GitHub Action, these might not be strictly necessary
# but are kept for consistency and potential local runs.
RAW_TRANSCRIPT_DIR = "data/raw_transcripts/"
RAW_STATIC_DIR = "data/raw_static/"
RAW_PDF_DIR = "data/raw_pdfs/" # For downloaded PDFs
# Use system temp directory for portability (still /tmp on Linux)
TEMP_DIR = os.path.join(tempfile.gettempdir(), "harvester_downloads")

# Cross-run de-duplication: skip URLs that already exist in DB
# Set DEDUP_BY_URL=false to allow cross-listed re-embeds per class
DEDUP_BY_URL = os.environ.get("DEDUP_BY_URL", "true").lower() in ("1", "true", "yes")

# Ensure necessary directories exist
os.makedirs("logs/error_screenshots", exist_ok=True)
os.makedirs(RAW_TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(RAW_STATIC_DIR, exist_ok=True)
os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def log_progress_bar(current: int, total: int, prefix: str = "Progress"):
    """Simple text-based progress indicator."""
    if total == 0:
        return
    percent = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '-' * (bar_length - filled)
    logging.info(f"{prefix}: |{bar}| {current}/{total} ({percent:.1f}%)")


def process_single_resource(driver: webdriver.Chrome, url: str, title: str, date_obj: datetime.datetime | None, class_name: str, section_tag: str):
    """
    Handles fetching, processing, cleaning, and embedding for a single resource URL.
    Determines content type and calls appropriate processing functions.
    """
    logging.info(f"Processing resource: {title} ({url})")
    raw_content_data = None # Store path (PDF) or text (HTML/Transcript)
    temp_pdf_path: str | None = None  # Track temp PDF path for reliable cleanup
    metadata_base = {
        "class_name": class_name,
        "source_url": url, # Store original URL
        "title": title,
        "retrieval_date": datetime.datetime.now().isoformat(),
        "section": section_tag,
    }
    content_type_tag = "unknown" # To be determined

    try:
        # 1. Determine Content Type and Fetch Raw Content
        # Prioritize known transcript platforms before generic content-type checks
        if "drive.google.com" in url or "zoom.us" in url:
            content_type_tag = "recording_transcript"
            logging.info("   Content type: Recording Transcript. Scraping...")
            transcript_text = scraping.scrape_transcript_from_url(driver, url)
            if transcript_text:
                raw_content_data = transcript_text
            else:
                logging.warning(f"   No transcript content scraped from {url}")
                return
        else:
            content_type_header = scraping.check_url_content_type(url)

            if content_type_header and "application/pdf" in content_type_header:
                content_type_tag = "pdf"
                logging.info("   Content type: PDF. Downloading...")
                safe_filename_base = utils.create_safe_filename(
                    f"{class_name}_{date_obj.strftime('%Y-%m-%d') if date_obj else 'nodate'}_{title}"
                )
                temp_pdf_path = scraping.download_file(url, TEMP_DIR, f"{safe_filename_base}.pdf")
                if temp_pdf_path:
                    raw_content_data = temp_pdf_path
                else:
                    raise ValueError("PDF download failed.")

            elif content_type_header and "text/html" in content_type_header:
                content_type_tag = "webpage"
                logging.info("   Content type: HTML. Scraping text...")
                html_text = scraping.scrape_html_content(url)
                if html_text:
                    raw_content_data = html_text
                else:
                    raise ValueError("HTML scraping yielded no content.")

            else:
                logging.warning(
                    f"   Unsupported or unknown content type '{content_type_header}' for {url}. Skipping."
                )
                return

        # --- Refinery Steps ---
        if raw_content_data:
            if content_type_tag == "pdf":
                # Process PDF page by page with guaranteed cleanup
                target_pdf_path = temp_pdf_path or (raw_content_data if isinstance(raw_content_data, str) else None)
                if not target_pdf_path:
                    raise ValueError("No PDF path available for processing.")

                logging.info(f"   Processing PDF content from: {target_pdf_path}")
                try:
                    pages_data = pdf_processing.process_pdf(target_pdf_path)  # Takes path
                    for page_data in pages_data:
                        # Combine text & image descriptions
                        combined_text = page_data.get("text", "")
                        image_descriptions = [
                            img.get("description", "")
                            for img in page_data.get("images", [])
                            if img.get("description")
                            and "unavailable" not in img.get("description", "").lower()
                            and "error" not in img.get("description", "").lower()
                        ]
                        if image_descriptions:
                            combined_text += "\n\n[Image Descriptions:]\n" + "\n".join(f"- {desc}" for desc in image_descriptions)

                        if len(combined_text) < 50:
                            continue  # Skip empty pages

                        metadata = {
                            **metadata_base,
                            "content_type": "pdf_page",
                            "source_file": os.path.basename(target_pdf_path),
                            "page_number": page_data.get("page_number"),
                            "links": page_data.get("links", []),
                        }
                        metadata = {k: v for k, v in metadata.items() if v is not None}

                        logging.info(f"   Embedding PDF page {metadata.get('page_number')}...")
                        embedding.chunk_and_embed_text(combined_text, metadata)
                finally:
                    # Ensure temp PDF is deleted regardless of success/failure
                    if target_pdf_path and os.path.exists(target_pdf_path):
                        try:
                            os.remove(target_pdf_path)
                        except OSError as e:
                            logging.warning(f"   Could not delete temp PDF {target_pdf_path}: {e}")

            elif content_type_tag in ["webpage", "recording_transcript"]:
                # Process extracted text (HTML or Transcript)
                logging.info(f"   Processing extracted text ({content_type_tag})...")
                # Clean the text using LLM
                clean_text = cleaning.clean_transcript_with_llm(raw_content_data) # Use same cleaner for now
                if not clean_text: raise ValueError("Cleaning returned empty text.")

                metadata = {
                    **metadata_base,
                    "content_type": content_type_tag,
                    "lecture_date": date_obj.strftime('%Y-%m-%d') if date_obj and content_type_tag == "recording_transcript" else None,
                }
                metadata = {k: v for k, v in metadata.items() if v is not None}

                logging.info(f"   Embedding extracted {content_type_tag} content...")
                embedding.chunk_and_embed_text(clean_text, metadata)

            else:
                logging.warning(f"   No processing logic for content_tag: {content_type_tag}")


    except Exception as process_err:
        logging.error(f"❌ Failed to process resource {title} ({url}): {process_err}")
        # Clean up temp file if it exists and processing failed
        if isinstance(raw_content_data, str) and content_type_tag == "pdf" and os.path.exists(raw_content_data):
            try: os.remove(raw_content_data)
            except OSError as e: logging.warning(f"   Could not delete failed temp PDF {raw_content_data}: {e}")


def main_pipeline(mode="daily"):
    """Main orchestration with batch processing."""
    logging.info(f"🚀 Hybrid Pipeline Started (Mode: {mode})")
    navigation.reset_course_tracking()
    
    # Determine cutoff date based on mode
    if mode == "daily":
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
        logging.info(f"Processing resources from the last 24 hours (since {cutoff_date.isoformat()})")
    elif mode == "backlog":
        cutoff_date = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        logging.info("Processing all historical resources (backlog mode).")
    else:
        raise ValueError(f"Unsupported pipeline mode: {mode}")
    
    # Configuration
    batch_size = getattr(config.SETTINGS, "resource_batch_size", 50)
    logging.info(f"Batch processing enabled: {batch_size} resources per batch")
    
    try:
        with navigation.launch_and_login() as driver:
            # Caches remain at pipeline scope
            recent_check_cache: dict[str, bool] = {}
            db_url_exists_cache: dict[str, bool] = {}
            seen_urls: set[str] = set()
            
            # Statistics tracking
            stats = {
                "courses_attempted": 0,
                "courses_successful": 0,
                "resources_discovered": 0,
                "resources_processed": 0,
                "resources_failed": 0,
            }
            
            logging.info("\n--- Starting Navigation Phase ---")
            
            # Optional: filter which courses to run via env var (comma-separated codes)
            course_filter_env = os.environ.get("COURSE_CODES") or os.environ.get("COURSE_FILTER")
            if course_filter_env:
                selected_codes = [code.strip() for code in course_filter_env.split(",") if code.strip()]
                course_items = [
                    (code, config.COURSE_MAP[code]) for code in selected_codes if code in config.COURSE_MAP
                ]
                missing = [code for code in selected_codes if code not in config.COURSE_MAP]
                if missing:
                    logging.warning(f"Requested course codes not in COURSE_MAP and will be skipped: {missing}")
            else:
                course_items = list(config.COURSE_MAP.items())

            # Auto-detect available courses on the page and filter out missing ones
            try:
                available_codes = navigation.get_available_course_codes(driver)
                if available_codes:
                    before = len(course_items)
                    course_items = [(c, d) for (c, d) in course_items if c in available_codes]
                    after = len(course_items)
                    skipped = sorted(set(code for code, _ in config.COURSE_MAP.items()) - set(available_codes))
                    logging.info(f"Filtering courses by availability: {before} -> {after}")
                    if skipped:
                        logging.warning(f"Skipping courses not present on page: {skipped}")
                else:
                    logging.warning("Could not detect available courses; proceeding with full COURSE_MAP.")
            except Exception as detect_err:
                logging.warning(f"Failed to detect available courses: {detect_err}")
            
            # Process courses one at a time (lighter memory footprint)
            for course_code, course_details in course_items:
                class_name = course_details["name"]
                group_name = course_details.get("group")
                stats["courses_attempted"] += 1
                
                logging.info(f"\n{'='*60}")
                logging.info(f"Course {stats['courses_attempted']}/{len(course_items)}: {class_name} ({course_code})")
                logging.info(f"{'='*60}")
                
                # Batch for this course
                course_resources = []
                
                try:
                    navigation.find_and_click_course_link(driver, course_code, group_name)
                    
                    if navigation.navigate_to_resources_section(driver):
                        sections = [
                            ("pre_read", config.PRE_READ_SECTION_TITLE),
                            ("in_class", config.IN_CLASS_SECTION_TITLE),
                            ("post_class", config.POST_CLASS_SECTION_TITLE),
                            ("sessions", config.SESSION_RECORDINGS_SECTION_TITLE),
                        ]
                        
                        for section_tag, title_text in sections:
                            logging.info(f"--- Section: {section_tag} ---")
                            try:
                                container_xpath, items = navigation.expand_section_and_get_items(driver, title_text)
                                logging.info(f"   Found {len(items)} items")
                                
                                for idx in range(len(items)):
                                    url, title, date_text = None, None, None
                                    try:
                                        # Re-find current item by index to avoid staleness during iteration
                                        item = driver.find_element(By.XPATH, f"{container_xpath}//div[contains(@class,'fileBox')][{idx+1}]")
                                        # Link: handle both descendant and ancestor <a> structures
                                        href = None
                                        link_el = None
                                        try:
                                            # Common case: anchor wraps the contents (ancestor of fileBox)
                                            link_el = item.find_element(By.XPATH, ".//ancestor::a[1]")
                                        except Exception:
                                            try:
                                                # Fallback: anchor inside the fileBox
                                                link_el = item.find_element(By.TAG_NAME, "a")
                                            except Exception:
                                                link_el = None

                                        if link_el is None:
                                            # Last resort: locate the Nth anchor that contains a fileBox under the same container
                                            try:
                                                link_el = driver.find_element(
                                                    By.XPATH,
                                                    f"({container_xpath}//a[.//div[contains(@class,'fileBox')]])[{idx+1}]"
                                                )
                                            except Exception:
                                                link_el = None

                                        if link_el is not None:
                                            href = link_el.get_attribute("href")
                                            if href and not href.startswith("http"):
                                                href = config.BASE_URL + href.lstrip('/')
                                            url = href
                                            # Title
                                            try:
                                                title_el = item.find_element(By.CSS_SELECTOR, config.RESOURCE_TITLE_CSS)
                                                title = title_el.text
                                            except Exception:
                                                title = url or ""
                                            # Date
                                            try:
                                                date_el = item.find_element(By.CSS_SELECTOR, config.RESOURCE_DATE_CSS)
                                                date_text = date_el.text
                                            except Exception:
                                                date_text = None

                                            if not url or not title:
                                                logging.info("      Skipping item (missing URL or Title)")
                                                continue

                                            if "youtube.com" in url or "youtu.be" in url:
                                                logging.info(f"      Skipping YouTube video: {title}")
                                                continue

                                            parsed_date = utils.parse_general_date(date_text) if date_text else None
                                            should_process = False
                                            if parsed_date:
                                                if parsed_date >= cutoff_date:
                                                    should_process = True
                                                else:
                                                    logging.info(f"      Skipping old resource (date: {parsed_date.strftime('%Y-%m-%d')})")
                                            else:
                                                exists_recently = recent_check_cache.get(url)
                                                if exists_recently is None:
                                                    exists_recently = embedding.check_if_embedded_recently_sync({"source_url": url}, days=2)
                                                recent_check_cache[url] = exists_recently
                                                should_process = not exists_recently
                                                if not should_process:
                                                    logging.info(f"      Skipping undated resource (found in recent DB): {url}")

                                            if should_process:
                                                if DEDUP_BY_URL:
                                                    exists_in_db = db_url_exists_cache.get(url)
                                                    if exists_in_db is None:
                                                        # We only have async version; best-effort skip extra DB roundtrip here
                                                        exists_in_db = embedding.url_exists_in_db_sync(url)
                                                    db_url_exists_cache[url] = exists_in_db
                                                    if exists_in_db:
                                                        logging.info(f"      Duplicate URL already in DB, skipping: {url}")
                                                        continue

                                                if url in seen_urls:
                                                    logging.info(f"      Duplicate URL (this run), skipping: {url}")
                                                else:
                                                    course_resources.append((url, title, parsed_date, class_name, section_tag))
                                                    seen_urls.add(url)
                                                    stats["resources_discovered"] += 1
                                                        
                                    except Exception as item_err:
                                        logging.warning(f"      Could not process one item in {section_tag}: {item_err}")
                                        continue
                                        
                            except Exception as section_err:
                                logging.warning(f"   Section '{section_tag}' failed: {section_err}")
                                continue
                        
                        stats["courses_successful"] += 1
                        
                except Exception as course_err:
                    logging.error(f"❌ Course {class_name} failed: {course_err}")
                    try:
                        driver.get(config.COURSES_URL)
                    except Exception:
                        pass
                    continue
                
                # Process this course's resources in batches
                if course_resources:
                    logging.info(f"\n--- Processing {len(course_resources)} resources for {class_name} ---")
                    
                    for batch_start in range(0, len(course_resources), batch_size):
                        batch_end = min(batch_start + batch_size, len(course_resources))
                        batch = course_resources[batch_start:batch_end]
                        
                        logging.info(f"Batch {batch_start//batch_size + 1}: Processing resources {batch_start+1}-{batch_end}/{len(course_resources)}")
                        
                        for url, title, date_obj, class_name_item, section_tag in batch:
                            try:
                                process_single_resource(driver, url, title, date_obj, class_name_item, section_tag)
                                stats["resources_processed"] += 1
                            except Exception as e:
                                logging.error(f"❌ Failed to process {title}: {e}")
                                stats["resources_failed"] += 1
                        
                        # Log progress
                        log_progress_bar(stats["resources_processed"] + stats["resources_failed"], 
                                       stats["resources_discovered"], 
                                       "Overall Progress")
            
            # Final statistics
            logging.info("\n" + "="*60)
            logging.info("PIPELINE STATISTICS")
            logging.info("="*60)
            for key, value in stats.items():
                logging.info(f"{key.replace('_', ' ').title()}: {value}")
            logging.info("="*60)
            
    except Exception as pipeline_err:
        logging.critical(f"Pipeline failed: {pipeline_err}", exc_info=True)
    finally:
        logging.info("🚀 Pipeline Finished")

# --- Allow running the script directly ---
if __name__ == "__main__":
    # --- Argument Parsing for Mode (Optional) ---
    # import argparse
    # parser = argparse.ArgumentParser(description="Run the Harvester & Refinery Pipeline.")
    # parser.add_argument('--mode', type=str, default="daily", choices=['daily', 'backlog'], help="Pipeline mode: 'daily' or 'backlog'.")
    # args = parser.parse_args()
    # --- Default Run (e.g., for GitHub Actions) ---
    pipeline_mode = os.environ.get("PIPELINE_MODE", "daily").lower()
    if pipeline_mode not in ["daily", "backlog"]:
        logging.warning(f"Invalid PIPELINE_MODE '{pipeline_mode}'. Defaulting to 'daily'.")
        pipeline_mode = "daily"

    main_pipeline(mode=pipeline_mode)