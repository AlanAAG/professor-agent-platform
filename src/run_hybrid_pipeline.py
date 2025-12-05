# src/run_hybrid_pipeline.py

import os
import datetime
import logging
import tempfile
import json
import time
import argparse
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# --- Import project modules (package-qualified for -m execution) ---
from src.harvester import navigation, scraping, config
from src.harvester.scraping import classify_url
from src.refinery import cleaning, embedding, pdf_processing, document_processing
from src.refinery.recording_processor import extract_transcript
from src.shared import utils


def sanitize_pipeline_event(event, hint):
    """Sanitize sensitive data from pipeline events."""
    sensitive_patterns = [
        "GEMINI_API_KEY",
        "SUPABASE_KEY",
        "COACH_PASSWORD",
        "SECRET_API_KEY",
        "OPENAI_API_KEY",
    ]

    def contains_sensitive(text: str) -> bool:
        upper_text = text.upper()
        return any(pattern in upper_text for pattern in sensitive_patterns)

    def sanitize(value):
        if isinstance(value, dict):
            for key, val in list(value.items()):
                key_str = str(key)
                if contains_sensitive(key_str):
                    value[key] = "[REDACTED]"
                    continue
                if isinstance(val, str) and contains_sensitive(val):
                    value[key] = "[REDACTED]"
                else:
                    sanitize(val)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, str) and contains_sensitive(item):
                    value[idx] = "[REDACTED]"
                else:
                    sanitize(item)

    sanitize(event)
    return event

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

SENTRY_DSN = os.getenv("SENTRY_DSN")
SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", "production")
SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))

if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        environment=SENTRY_ENVIRONMENT,
        traces_sample_rate=SENTRY_TRACES_SAMPLE_RATE,
        integrations=[LoggingIntegration(level=logging.INFO, event_level=logging.ERROR)],
        before_send=sanitize_pipeline_event,
        profiles_sample_rate=0.1,
    )
    logging.info("Sentry monitoring initialized for hybrid pipeline")

# --- Configuration ---
# Directories for saving raw files locally (primarily for testing/backup)
RAW_TRANSCRIPT_DIR = "data/raw_transcripts/"
RAW_STATIC_DIR = "data/raw_static/"
RAW_PDF_DIR = "data/raw_pdfs/" # For downloaded PDFs
# Use system temp directory for portability (still /tmp on Linux)
TEMP_DIR = os.path.join(tempfile.gettempdir(), "harvester_downloads")

# Cross-run de-duplication: skip URLs that already exist in DB
# Set DEDUP_BY_URL=false to allow cross-listed re-embeds per class
DEDUP_BY_URL = os.environ.get("DEDUP_BY_URL", "true").lower() in ("1", "true", "yes")

OFFICE_VIEWER_CONTENT_TYPES = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-powerpoint",
    "application/vnd.ms-excel",
    "application/msword",
    "application/vnd.ms-powerpoint.presentation.macroenabled.12",
    "application/vnd.ms-excel.sheet.macroenabled.12",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
)

# Ensure necessary directories exist
os.makedirs("logs/error_screenshots", exist_ok=True)
os.makedirs(RAW_TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(RAW_STATIC_DIR, exist_ok=True)
os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def save_run_summary(
    stats: Dict[str, Any],
    mode: str,
    start_time: datetime.datetime,
    errors: List[Dict[str, str]],
):
    """Save pipeline run summary to JSON file."""
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    summary = {
        "run_metadata": {
            "timestamp": start_time.isoformat(),
            "mode": mode,
            "duration_seconds": round(duration, 2),
            "duration_human": f"{int(duration // 60)}m {int(duration % 60)}s",
            "success": stats.get("resources_failed", 0) == 0,
        },
        "statistics": stats,
        "errors": errors[:50],  # Limit to first 50 errors
        "error_count": len(errors),
    }

    # Save to timestamped file
    filename = f"logs/run_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("logs", exist_ok=True)

    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)

    # Also save as "latest" for easy access
    with open("logs/run_summary_latest.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"📊 Run summary saved to {filename}")

    # Print summary to console
    logging.info("\n" + "=" * 60)
    logging.info("RUN SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Mode: {mode}")
    logging.info(f"Duration: {summary['run_metadata']['duration_human']}")
    logging.info(
        f"Courses: {stats.get('courses_successful', 0)}/{stats.get('courses_attempted', 0)} successful"
    )
    logging.info(
        f"Resources: {stats.get('resources_processed', 0)} processed, {stats.get('resources_failed', 0)} failed"
    )
    if errors:
        logging.info(
            f"Errors: {len(errors)} occurred (see {filename} for details)"
        )
    logging.info("=" * 60)


def process_single_resource(
    driver: webdriver.Chrome,
    url: str,
    title: str,
    date_obj: datetime.datetime | None,
    class_name: str,
    section_tag: str,
    stats: Dict[str, Any],
    table_name: str,
):
    """
    Handles fetching, processing, cleaning, and embedding for a single resource URL.
    Determines content type and calls appropriate processing functions.
    """
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to ensure temp directory {TEMP_DIR}: {e}")
        raise

    if not os.access(TEMP_DIR, os.W_OK):
        logging.error(f"Temp directory {TEMP_DIR} is not writable.")
        raise PermissionError(f"Temp directory {TEMP_DIR} is not writable.")

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
            logging.info("   Potential recording detected. Attempting transcript extraction...")
            resource_type = classify_url(url)
            transcript_text = extract_transcript(
                driver,
                url,
                {
                    "RECORDING_ZOOM": "ZOOM_RECORDING",
                    "RECORDING_DRIVE": "DRIVE_RECORDING",
                }.get(resource_type, resource_type),
            )
            if transcript_text:
                content_type_tag = "recording_transcript"
                raw_content_data = transcript_text
                # Telemetry: count successfully scraped transcripts
                try:
                    stats["zoom_transcripts_scraped"] = stats.get("zoom_transcripts_scraped", 0) + 1
                except Exception:
                    pass
            else:
                logging.info(f"   No transcript content scraped from {url}. Checking content type...")

        # If no transcript found (or not a recording URL), proceed to check content type
        if not raw_content_data:
            content_type_header = scraping.check_url_content_type(url)

            if content_type_header and "application/pdf" in content_type_header:
                content_type_tag = "pdf"
                logging.info("   Content type: PDF. Downloading...")
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
                logging.info("   Content type: HTML. Scraping text...")
                html_text = scraping.scrape_html_content(url, driver=driver)
                if html_text:
                    raw_content_data = html_text
                else:
                    raise ValueError("HTML scraping yielded no content.")

            elif content_type_header and any(
                mime in content_type_header for mime in OFFICE_VIEWER_CONTENT_TYPES
            ):
                content_type_tag = "office_document"
                logging.info("   Content type: Office document. Extracting via viewer...")
                if not driver:
                    raise ValueError("Office document extraction requires an authenticated driver.")
                document_text = document_processing.extract_drive_content(
                    driver, url, "office_document"
                )
                if document_text:
                    raw_content_data = document_text
                else:
                    logging.warning("   Viewer extraction failed or empty. Attempting direct download fallback...")
                    # Fallback: Download and process locally
                    try:
                        # Extract extension from URL or header if possible, else rely on what download_file does or rename
                        # scraping.download_file uses the filename provided.
                        # We need to guess extension.
                        filename_from_url = url.split('/')[-1].split('?')[0]
                        if not any(filename_from_url.endswith(ext) for ext in ['.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls']):
                             # Try to infer from content_type_header if no extension in URL
                             if "word" in content_type_header:
                                 filename_from_url = "downloaded_doc.docx"
                             elif "presentation" in content_type_header or "powerpoint" in content_type_header:
                                 filename_from_url = "downloaded_pres.pptx"
                             elif "spreadsheet" in content_type_header or "excel" in content_type_header:
                                 filename_from_url = "downloaded_sheet.xlsx"
                             else:
                                 filename_from_url = "downloaded_file.bin" # risky

                        temp_office_path = scraping.download_file(url, TEMP_DIR, filename_from_url)
                        if temp_office_path:
                            try:
                                local_text = document_processing.process_local_office_file(temp_office_path)
                                if local_text:
                                    raw_content_data = local_text
                                    logging.info(f"   Successfully extracted text from local office file ({len(local_text)} chars).")
                                else:
                                     logging.warning("   Local office extraction yielded no text.")
                            finally:
                                if os.path.exists(temp_office_path):
                                    os.remove(temp_office_path)
                                    logging.debug(f"   Cleaned up temp office file: {temp_office_path}")

                        if not raw_content_data:
                             raise ValueError("Both viewer extraction and download fallback failed.")

                    except Exception as fallback_err:
                        logging.error(f"   Download fallback failed: {fallback_err}")
                        raise ValueError("Office document extraction yielded no content (viewer and fallback failed).")

            else:
                logging.warning(
                    f"   Unsupported or unknown content type '{content_type_header}' for {url}. Skipping."
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
                    page_entries: List[Dict[str, Any]] = []

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

                        page_entries.append(
                            {
                                "text": combined_text,
                                "metadata": {
                                    **metadata_base,
                                    "content_type": "pdf_page",
                                    "source_file": os.path.basename(target_pdf_path),
                                    "page_number": page_data.get("page_number"),
                                    "links": page_data.get("links", []),
                                },
                            }
                        )

                    if not page_entries:
                        logging.info("   Skipping embedding: no PDF pages with sufficient content.")
                        return True

                    aggregated_text = "\n\n".join(entry["text"] for entry in page_entries)
                    content_hash = utils.calculate_content_hash(aggregated_text)

                    has_changed = True
                    if DEDUP_BY_URL:
                        has_changed = embedding.content_has_changed_sync(url, content_hash, table_name=table_name)

                    if not has_changed:
                        logging.info(f"   Content unchanged for {url}; skipping embedding.")
                        return False

                    embedding.delete_documents_by_source_url(url, table_name=table_name)
                    logging.info(f"   Embedding updated PDF content (hash={content_hash[:12]}…)")

                    for entry in page_entries:
                        metadata = {**entry["metadata"], "content_hash": content_hash}
                        metadata = {k: v for k, v in metadata.items() if v is not None}
                        logging.info(f"   Embedding PDF page {metadata.get('page_number')}...")
                        embedding.chunk_and_embed_text(entry["text"], metadata, table_name=table_name)
                    # Telemetry: count successfully processed PDF documents (once per file)
                    stats["pdf_documents_processed"] = stats.get("pdf_documents_processed", 0) + 1

                finally:
                    # Ensure temp PDF is deleted regardless of success/failure
                    if temp_pdf_path and isinstance(temp_pdf_path, str):
                        try:
                            if os.path.exists(temp_pdf_path):
                                os.remove(temp_pdf_path)
                                logging.debug(f"Cleaned up temp PDF: {temp_pdf_path}")
                        except OSError as e:
                            logging.warning(f"Could not delete temp PDF {temp_pdf_path}: {e}")

            elif content_type_tag in ["webpage", "recording_transcript", "office_document"]:
                # Process extracted text (HTML or Transcript)
                logging.info(f"   Processing extracted text ({content_type_tag})...")
                # Clean the text using LLM
                clean_text = cleaning.clean_transcript_with_llm(raw_content_data)  # Use same cleaner for now
                if not clean_text:
                    raise ValueError("Cleaning returned empty text.")

                content_hash = utils.calculate_content_hash(clean_text)

                has_changed = True
                if DEDUP_BY_URL:
                    has_changed = embedding.content_has_changed_sync(url, content_hash, table_name=table_name)

                if not has_changed:
                    logging.info(f"   Content unchanged for {url}; skipping embedding.")
                    return False

                embedding.delete_documents_by_source_url(url, table_name=table_name)

                metadata = {
                    **metadata_base,
                    "content_type": content_type_tag,
                    "lecture_date": date_obj.strftime('%Y-%m-%d') if date_obj and content_type_tag == "recording_transcript" else None,
                    "content_hash": content_hash,
                }
                metadata = {k: v for k, v in metadata.items() if v is not None}

                logging.info(f"   Embedding extracted {content_type_tag} content (hash={content_hash[:12]}…)")
                embedding.chunk_and_embed_text(clean_text, metadata, table_name=table_name)
                if content_type_tag == "office_document":
                    stats["office_documents_processed"] = stats.get("office_documents_processed", 0) + 1

            else:
                logging.warning(f"   No processing logic for content_tag: {content_type_tag}")


        # Success
        return True

    except Exception as process_err:
        logging.error(f"❌ Failed to process resource {title} ({url}): {process_err}")
        # Clean up temp file if it exists and processing failed
        if isinstance(raw_content_data, str) and content_type_tag == "pdf" and os.path.exists(raw_content_data):
            try:
                os.remove(raw_content_data)
            except OSError as e:
                logging.warning(f"   Could not delete failed temp PDF {raw_content_data}: {e}")
        # Re-raise so caller can track stats and errors
        raise


def main_pipeline(mode="daily", cohort_id="2029"):
    """Main orchestration with monitoring and summary generation."""
    with sentry_sdk.start_transaction(op="pipeline", name=f"pipeline_{mode}"):
        start_time = datetime.datetime.now()
        start_epoch = time.time()
        logging.info(
            f"🚀 Hybrid Pipeline Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} (Mode: {mode}, Cohort: {cohort_id})"
        )
        
        # --- Load Cohort Configuration ---
        if cohort_id not in config.COHORTS:
            raise ValueError(f"Invalid cohort ID: {cohort_id}. Available: {list(config.COHORTS.keys())}")

        cohort_config = config.COHORTS[cohort_id]
        table_name = cohort_config["table_name"]
        course_map = cohort_config["course_map"]
        cohort_selectors = cohort_config.get("selectors", {})

        # --- Set Cohort Credentials in Env ---
        # The navigation module (perform_login) looks for COACH_USERNAME/COACH_PASSWORD
        user_env_key, pass_env_key = cohort_config["credentials"]

        logging.info(f"Authenticating using credentials from: {user_env_key}")

        if not os.getenv(user_env_key):
            raise ValueError(f"Missing environment variable: {user_env_key}. Cannot log in for cohort {cohort_id}.")
        if not os.getenv(pass_env_key):
            raise ValueError(f"Missing environment variable: {pass_env_key}. Cannot log in for cohort {cohort_id}.")

        os.environ["COACH_USERNAME"] = os.environ[user_env_key]
        os.environ["COACH_PASSWORD"] = os.environ[pass_env_key]

        logging.info(f"Loaded configuration for {cohort_config['name']} (Table: {table_name})")

        # --- Inject Cohort Selectors ---
        # Override config.RESOURCES_TAB_SELECTORS based on cohort config
        if "resources_tab" in cohort_selectors:
            # config.RESOURCES_TAB_SELECTORS is a list of tuples.
            # The selector from cohort_config might be a single tuple or string.
            res_tab_selector = cohort_selectors["resources_tab"]
            if isinstance(res_tab_selector, tuple):
                 config.RESOURCES_TAB_SELECTORS = [res_tab_selector]
            elif isinstance(res_tab_selector, list):
                 config.RESOURCES_TAB_SELECTORS = res_tab_selector

        # Override config.COURSE_CARD_FALLBACK_XPATH_TEMPLATE based on cohort config
        if "course_card" in cohort_selectors:
            course_card_selector = cohort_selectors["course_card"]
            if isinstance(course_card_selector, tuple):
                # Assuming the tuple contains (By.XPATH, "selector_string")
                # We extract the selector string to update the template variable
                config.COURSE_CARD_FALLBACK_XPATH_TEMPLATE = course_card_selector[1]
            elif isinstance(course_card_selector, str):
                config.COURSE_CARD_FALLBACK_XPATH_TEMPLATE = course_card_selector


        # Reset course links seen in navigation to handle duplicates correctly.
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

        # Initialize stats & errors
        stats: Dict[str, Any] = {
            "courses_attempted": 0,
            "courses_successful": 0,
            "resources_discovered": 0,
            # Telemetry metrics
            "resources_found": 0,
            "zoom_transcripts_scraped": 0,
            "pdf_documents_processed": 0,
            "resources_processed": 0,
            "resources_failed": 0,
        }

        errors: List[Dict[str, str]] = []
        
        # Tracks URLs attempted in this run to avoid duplicate processing attempts
        attempted_resource_urls: set[str] = set()
        # Tracks URLs successfully processed in this run for reporting
        successfully_processed_urls: set[str] = set()

        try:
            with navigation.launch_and_login() as driver:
                # Caches remain at pipeline scope
                recent_check_cache: dict[str, bool] = {}
                # seen_urls is now replaced by processed_resource_urls set for de-duplication

                # --- Navigation and Resource Discovery ---
                
                logging.info("\n--- Starting Navigation Phase ---")
                
                # Course Filtering Logic (Retained from existing code)
                course_filter_env = os.environ.get("COURSE_CODES") or os.environ.get("COURSE_FILTER")
                if course_filter_env:
                    selected_codes = [code.strip() for code in course_filter_env.split(",") if code.strip()]
                    course_items = [
                        (code, course_map[code]) for code in selected_codes if code in course_map
                    ]
                else:
                    course_items = list(course_map.items())

                # Update course attempts after filtering
                stats["courses_attempted"] = len(course_items)

                # Process courses one at a time (lighter memory footprint)
                for course_code, course_details in course_items:
                    class_name = course_details["name"]
                    group_name = course_details.get("group")
                    
                    logging.info(f"\n{'='*60}")
                    logging.info(f"Course: {class_name} ({course_code})")
                    logging.info(f"{'='*60}")
                    
                    # Batch for this course (No longer needed to store all resources in one list, processing happens immediately)
                    
                    try:
                        # Navigation will check internal cache and skip if already navigated in this run.
                        try:
                            navigated = navigation.find_and_click_course_link(driver, course_code, group_name)
                        except navigation.CourseNavigationError as nav_err:
                            logging.warning(
                                "Skipping course %s (%s): %s",
                                course_code,
                                class_name,
                                nav_err,
                            )
                            try:
                                driver.get(config.COURSES_URL)
                            except Exception:
                                pass
                            continue
                        except (TimeoutException, NoSuchElementException) as nav_err:
                            logging.warning(
                                "Skipping course %s (%s): navigation failed (%s).",
                                course_code,
                                class_name,
                                nav_err,
                            )
                            try:
                                driver.get(config.COURSES_URL)
                            except Exception:
                                pass
                            continue

                        if not navigated:
                            logging.info(
                                "Course %s (%s) was already processed earlier in this run. Skipping resource extraction.",
                                course_code,
                                class_name,
                            )
                            continue
                        
                        if not navigation.navigate_to_resources_section(driver):
                            logging.warning(
                                "Could not navigate to resources section for %s. Skipping.",
                                course_code,
                            )
                            continue

                        navigation._take_progress_screenshot(
                            driver, f"03b_on_resources_page_{course_code}"
                        )

                        sections = [
                            ("pre_read", config.PRE_READ_SECTION_TITLE),
                            ("in_class", config.IN_CLASS_SECTION_TITLE),
                            ("post_class", config.POST_CLASS_SECTION_TITLE),
                            ("sessions", config.SESSION_RECORDINGS_SECTION_TITLE),
                        ]

                        for section_tag, title_text in sections:
                            logging.info(f"--- Section: {section_tag} ---")
                            try:
                                container_descriptor, anchor_elements, header_el = navigation.expand_section_and_get_items(
                                    driver,
                                    title_text,
                                )

                                if header_el is None:
                                    logging.info(
                                        "   Skipping section '%s' (header not found or not expanded).",
                                        section_tag,
                                    )
                                    continue

                                try:
                                    # Telemetry: count items found at navigation stage
                                    try:
                                        stats["resources_found"] += len(anchor_elements)
                                    except Exception:
                                        pass
                                    logging.info(f"   Found {len(anchor_elements)} items")

                                    container_xpath_for_items = container_descriptor or ""
                                    if "::" in container_xpath_for_items:
                                        locator_type, locator_value = container_xpath_for_items.split("::", 1)
                                        if locator_type.strip().lower() == "xpath":
                                            container_xpath_for_items = locator_value
                                        else:
                                            container_xpath_for_items = ""

                                    file_box_elements: List[Any] = []
                                    if container_xpath_for_items:
                                        try:
                                            file_box_elements = driver.find_elements(
                                                By.XPATH,
                                                f"{container_xpath_for_items}//div[contains(@class,'fileBox')]",
                                            )
                                        except Exception:
                                            file_box_elements = []

                                    # --- START IMMEDIATE RESOURCE PROCESSING ---
                                    for idx, anchor in enumerate(anchor_elements):
                                        url, title, date_text = None, None, None
                                        try:
                                            item = None
                                            if idx < len(file_box_elements):
                                                item = file_box_elements[idx]

                                            if item is None:
                                                try:
                                                    item = anchor.find_element(
                                                        By.XPATH,
                                                        ".//ancestor::div[contains(@class,'fileBox')][1]",
                                                    )
                                                except Exception:
                                                    item = None

                                            if item is None:
                                                logging.info(
                                                    "      Skipping item (unable to resolve container element)"
                                                )
                                                continue

                                            href = ""
                                            try:
                                                href = (anchor.get_attribute("href") or "").strip()
                                            except Exception:
                                                href = ""

                                            link_el = anchor if href else None
                                            if not href:
                                                link_locators = [
                                                    (By.CSS_SELECTOR, "div.fileContentCol a[href]"),
                                                    (By.CSS_SELECTOR, "a[href]"),
                                                    (By.XPATH, ".//ancestor::a[1]"),
                                                ]
                                                for by, selector in link_locators:
                                                    try:
                                                        link_el = item.find_element(by, selector)
                                                        href = (link_el.get_attribute("href") or "").strip()
                                                        if href:
                                                            break
                                                    except Exception:
                                                        continue

                                            if not href:
                                                logging.info("      Skipping item (no link element found)")
                                                continue

                                            if href and not href.startswith("http"):
                                                href = config.BASE_URL + href.lstrip('/')
                                            url = href

                                            try:
                                                title_el = item.find_element(By.CSS_SELECTOR, config.RESOURCE_TITLE_CSS)
                                                title = title_el.text
                                            except Exception:
                                                title = url or ""

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

                                            # --- NEW LOGIC START ---
                                            # Check for Manually Ingested Transcript (Deduplication)
                                            # Assumption: Manual ingest stores key as "{class_name}.txt|{title}"
                                            # We check exact match, then normalized match (stripped, lowercase, replaced '&')

                                            potential_keys = [
                                                f"{class_name}.txt|{title}",
                                                f"{class_name}.txt|{title.strip()}",
                                                f"{class_name}.txt|{title.strip().lower()}",
                                                f"{class_name}.txt|{title.replace('&', 'and')}",
                                                f"{class_name}.txt|{title.strip().replace('&', 'and')}",
                                                f"{class_name}.txt|{title.strip().lower().replace('&', 'and')}",
                                            ]

                                            is_manually_ingested = False
                                            matched_key = ""

                                            for key_variant in potential_keys:
                                                try:
                                                    if embedding.url_exists_in_db_sync(key_variant, table_name=table_name):
                                                        is_manually_ingested = True
                                                        matched_key = key_variant
                                                        break
                                                except Exception as e:
                                                    logging.warning(f"      Manual key check failed for {key_variant}: {e}")
                                                    continue

                                            if is_manually_ingested:
                                                logging.info(f"      Skipping {title} (Matches manual ingest: {matched_key})")
                                                continue
                                            # --- NEW LOGIC END ---

                                            parsed_date = utils.parse_general_date(date_text) if date_text else None
                                            should_process = False

                                            # Filtering Logic (Date / Recent Check)
                                            if parsed_date:
                                                if parsed_date >= cutoff_date:
                                                    should_process = True
                                                else:
                                                    logging.info(
                                                        f"      Skipping old resource (date: {parsed_date.strftime('%Y-%m-%d')})"
                                                    )
                                            else:
                                                exists_recently = recent_check_cache.get(url)
                                                if exists_recently is None:
                                                    exists_recently = embedding.check_if_embedded_recently_sync(
                                                        {"source_url": url}, days=2, table_name=table_name
                                                    )
                                                recent_check_cache[url] = exists_recently
                                                should_process = not exists_recently
                                                if not should_process:
                                                    logging.info(
                                                        f"      Skipping undated resource (found in recent DB): {url}"
                                                    )

                                            if not should_process:
                                                continue

                                            if url in attempted_resource_urls:
                                                logging.info(f"      Duplicate URL (this run), skipping: {url}")
                                                continue

                                            stats["resources_discovered"] += 1

                                            # Mark URL as attempted to prevent duplicate processing attempts
                                            attempted_resource_urls.add(url)

                                            # --- PROCESS RESOURCE IMMEDIATELY ---
                                            logging.info(
                                                f"      ADDING resource to process queue: {title} (Section: {section_tag})"
                                            )

                                            try:
                                                success = process_single_resource(
                                                    driver,
                                                    url,
                                                    title,
                                                    parsed_date,
                                                    class_name,
                                                    section_tag,
                                                    stats,
                                                    table_name,
                                                )
                                                if success:
                                                    stats["resources_processed"] += 1
                                                    # Track URL as successfully processed for reporting
                                                    successfully_processed_urls.add(url)
                                            except Exception as e:
                                                logging.error(f"❌ Failed to process {title}: {e}")
                                                stats["resources_failed"] += 1
                                                errors.append(
                                                    {
                                                        "resource_title": title,
                                                        "resource_url": url,
                                                        "class_name": class_name,
                                                        "error_type": type(e).__name__,
                                                        "error_message": str(e),
                                                        "timestamp": datetime.datetime.now().isoformat(),
                                                    }
                                                )

                                        except Exception as item_err:
                                            logging.warning(
                                                f"      Could not process one item in {section_tag}: {item_err}"
                                            )
                                            continue

                                    # --- END IMMEDIATE RESOURCE PROCESSING ---

                                finally:
                                    try:
                                        logging.info("   Closing section '%s' to restore collapsed state.", section_tag)
                                        driver.execute_script("arguments[0].click();", header_el)
                                        time.sleep(1.0)
                                        navigation._take_progress_screenshot(
                                            driver,
                                            f"05_closed_section_{section_tag}_{course_code}",
                                        )
                                    except Exception as close_err:
                                        logging.debug(
                                            "   Failed to close section '%s' cleanly: %s",
                                            section_tag,
                                            close_err,
                                        )

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

        except Exception as pipeline_err:
            logging.critical(f"Pipeline failed with critical error: {pipeline_err}", exc_info=True)
            errors.append(
                {
                    "error_type": "CRITICAL_PIPELINE_FAILURE",
                    "error_message": str(pipeline_err),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )
            if SENTRY_DSN:
                with sentry_sdk.push_scope() as scope:
                    scope.set_context(
                        "pipeline",
                        {
                            "mode": mode,
                            "stats": stats,
                            "errors_count": len(errors),
                        },
                    )
                    scope.set_extra("exception_message", str(pipeline_err))
                    sentry_sdk.capture_exception(pipeline_err)
            raise
        finally:
            # Always save summary, even on failure
            try:
                save_run_summary(stats, mode, start_time, errors)
            except Exception as summary_err:
                logging.error(f"Failed to save run summary: {summary_err}")

            # Timer-based runtime and final metrics log
            total_runtime_sec = round(time.time() - start_epoch, 2)
            subjects_failed = max(stats.get("courses_attempted", 0) - stats.get("courses_successful", 0), 0)
            logging.info(
                (
                    "📈 Pipeline Metrics | runtime_s=%s resources_found=%s resources_processed=%s "
                    "zoom_transcripts_scraped=%s pdf_documents_processed=%s subjects_failed=%s"
                ),
                total_runtime_sec,
                stats.get("resources_found", 0),
                stats.get("resources_processed", 0),
                stats.get("zoom_transcripts_scraped", 0),
                stats.get("pdf_documents_processed", 0),
                subjects_failed,
            )

            final_status = (
                "SUCCESS" if (len(errors) == 0 and stats.get("resources_failed", 0) == 0) else "FAILURE"
            )

            # Persist concise status JSON for external monitors
            try:
                os.makedirs("data", exist_ok=True)
                concise_report = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "status": final_status,
                    "runtime_seconds": total_runtime_sec,
                    "metrics": {
                        "resources_found": stats.get("resources_found", 0),
                        "resources_processed": stats.get("resources_processed", 0),
                        "zoom_transcripts_scraped": stats.get("zoom_transcripts_scraped", 0),
                        "pdf_documents_processed": stats.get("pdf_documents_processed", 0),
                        "subjects_failed": subjects_failed,
                    },
                }
                report_path = getattr(config.SETTINGS, "metrics_report_path", "data/pipeline_status.json")
                with open(report_path, "w") as f:
                    json.dump(concise_report, f, indent=2)
                logging.info("📄 Pipeline status saved to %s", report_path)
            except Exception as e:
                logging.error("Failed to write pipeline status JSON: %s", e)

            if SENTRY_DSN:
                with sentry_sdk.push_scope() as scope:
                    scope.set_context(
                        "pipeline",
                        {
                            "mode": mode,
                            "stats": stats,
                            "errors_count": len(errors),
                            "duration_seconds": total_runtime_sec,
                            "final_status": final_status,
                        },
                    )
                    sentry_sdk.capture_message(
                        f"Pipeline {mode} completed",
                        level="info",
                    )

            logging.info("🚀 Hybrid Pipeline Finished.")


# --- Allow running the script directly ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professor Agent Harvester Pipeline")
    parser.add_argument("--mode", default=os.environ.get("PIPELINE_MODE", "daily").lower(), help="Pipeline mode (daily, backlog)")
    parser.add_argument("--cohort", default="2029", help="Cohort ID (2028, 2029)")

    args = parser.parse_args()

    pipeline_mode = args.mode
    if pipeline_mode not in ["daily", "backlog"]:
        logging.warning(f"Invalid PIPELINE_MODE '{pipeline_mode}'. Defaulting to 'daily'.")
        pipeline_mode = "daily"

    main_pipeline(mode=pipeline_mode, cohort_id=args.cohort)