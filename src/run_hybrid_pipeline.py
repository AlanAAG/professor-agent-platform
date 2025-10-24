# src/run_hybrid_pipeline.py

import os
import asyncio
import datetime
import logging
import tempfile
from playwright.async_api import async_playwright
from dotenv import load_dotenv

# --- Import project modules (package-qualified for -m execution) ---
from src.harvester import navigation, scraping, config
from src.refinery import cleaning, embedding, pdf_processing  # Added pdf_processing
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


async def process_single_resource(context, url: str, title: str, date_obj: datetime.datetime | None, class_name: str, section_tag: str):
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
            transcript_text = await scraping.scrape_transcript_from_url(context, url)
            if transcript_text:
                raw_content_data = transcript_text
            else:
                logging.warning(f"   No transcript content scraped from {url}")
                return
        else:
            content_type_header = await scraping.check_url_content_type(url)

            if content_type_header and "application/pdf" in content_type_header:
                content_type_tag = "pdf"
                logging.info("   Content type: PDF. Downloading...")
                safe_filename_base = utils.create_safe_filename(
                    f"{class_name}_{date_obj.strftime('%Y-%m-%d') if date_obj else 'nodate'}_{title}"
                )
                temp_pdf_path = await scraping.download_file(url, TEMP_DIR, f"{safe_filename_base}.pdf")
                if temp_pdf_path:
                    raw_content_data = temp_pdf_path
                else:
                    raise ValueError("PDF download failed.")

            elif content_type_header and "text/html" in content_type_header:
                content_type_tag = "webpage"
                logging.info("   Content type: HTML. Scraping text...")
                html_text = await scraping.scrape_html_content(url)
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


async def main_pipeline(mode="daily"):
    """
    Main orchestration function for the stateless Harvester & Refinery pipeline.
    Connects scraping directly to cleaning and embedding.
    """
    logging.info(f"🚀 Hybrid Pipeline Started (Mode: {mode})")

    # Determine cutoff date based on mode
    if mode == "daily":
        # Process anything from the last 24 hours (rolling window)
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
        logging.info(f"Processing resources from the last 24 hours (since {cutoff_date.isoformat()})")
    elif mode == "backlog":
        cutoff_date = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc) # Get everything
        logging.info("Processing all historical resources (backlog mode).")
    else:
        raise ValueError(f"Unsupported pipeline mode: {mode}")

    browser = None
    context = None
    page = None
    try:
        async with async_playwright() as p:
            browser, context, page = await navigation.launch_and_login(p)
            if not page:
                logging.warning("Login failed or not configured. Falling back to unauthenticated browser for targeted test.")
                # Close any partially created contexts/browsers
                try:
                    if context:
                        await context.close()
                except Exception:
                    pass
                try:
                    if browser:
                        await browser.close()
                except Exception:
                    pass
                # Launch fresh browser without auth; sufficient for public Zoom links
                browser = await p.chromium.launch()
                context = await browser.new_context()
                page = await context.new_page()

            # Store resources found during navigation phase
            resources_to_process = [] # List of tuples: (url, title, date_obj | None, class_name, section_tag)
            # Caches to avoid redundant checks within a single run
            recent_check_cache: dict[str, bool] = {}
            seen_urls: set[str] = set()
            db_url_exists_cache: dict[str, bool] = {}

            # --- Navigation Phase: Collect all new/relevant resource links ---
            logging.info("\n--- Starting Navigation Phase ---")
            # for course_code, course_details in config.COURSE_MAP.items(): # Use COURSE_MAP from config
            #     class_name = course_details["name"]
            #     group_name = course_details.get("group") # Group might be None
            #
            #     logging.info(f"\n--- Checking Course: {class_name} ({course_code}) ---")
            #     try:
            #         # Navigate to the specific course page
            #         await navigation.find_and_click_course_link(page, course_code, group_name)
            #         current_course_url = page.url # Store URL to potentially return later
            #
            #         # --- Scrape Resources Tab (NEW LOGIC) ---
            #         if await navigation.navigate_to_resources_section(page):
            #             # Define the sections you want to scrape using your new config selectors
            #             # Section headers follow partner script pattern: div.dlvLeftHeader:has-text(...)
            #             # Item containers remain div.fileBox under the nearest following sibling container
            #             sections_to_scrape = [
            #                 ("pre_read", config.PRE_READ_SECTION_SELECTOR),
            #                 ("in_class", config.IN_CLASS_SECTION_SELECTOR),
            #                 ("post_class", config.POST_CLASS_SECTION_SELECTOR),
            #                 ("sessions", config.RECORDINGS_LINK_SELECTOR), # Re-use existing selector for sessions
            #             ]
            #
            #             for section_tag, header_selector in sections_to_scrape:
            #                 logging.info(f"--- Scraping Section: {section_tag} ---")
            #                 try:
            #                     section_header = page.locator(header_selector).first
            #                     await section_header.wait_for(state="visible", timeout=7000)
            #                     
            #                     # --- THIS IS THE FIX ---
            #                     # Click the section header to expand the accordion
            #                     logging.info(f"   Found section header: {section_tag}. Clicking to expand...")
            #                     await section_header.click(timeout=5000)
            #                     # Wait for the accordion animation to finish
            #                     await page.wait_for_timeout(1500) 
            #                     # --- END FIX ---
            #
            #                     # The header `div.sc-kRJjUj` sits inside `div.sc-hsNTtK`; items live in the next sibling container
            #                     section_container = section_header.locator("xpath=./parent::div/following-sibling::div[1]")
            #                     item_locators = section_container.locator(config.RESOURCE_ITEM_SELECTOR)
            #                     count = await item_locators.count()
            #
            #                     logging.info(f"   Found {count} items in section '{section_tag}'.")
            #
            #                     for i in range(count):
            #                         item = item_locators.nth(i)
            #                         url, title, date_text = None, None, None
            #                         try:
            #                             link_locator = item.locator("a").first
            #                             url = await link_locator.get_attribute("href", timeout=2000)
            #                             if url and not url.startswith("http"):
            #                                 url = config.BASE_URL + url.lstrip('/')
            #
            #                             title = await item.locator(config.RESOURCE_TITLE_SELECTOR).text_content(timeout=2000)
            #                             date_text_element = item.locator(config.RESOURCE_DATE_SELECTOR).first
            #                             date_text = await date_text_element.text_content(timeout=1000) if await date_text_element.is_visible() else None
            #
            #                             if not url or not title:
            #                                 logging.warning("      Skipping item (missing URL or Title)")
            #                                 continue
            #
            #                             # Skip YouTube links
            #                             if "youtube.com" in url or "youtu.be" in url:
            #                                 logging.info(f"      Skipping YouTube video: {title}")
            #                                 continue
            #
            #                             parsed_date = utils.parse_general_date(date_text) if date_text else None
            #
            #                             # Run date/dupe checks
            #                             should_process = False
            #                             if parsed_date:
            #                                 if parsed_date >= cutoff_date:
            #                                     should_process = True
            #                                 else:
            #                                     logging.info(f"      Skipping old resource (date: {parsed_date.strftime('%Y-%m-%d')})")
            #                             else:
            #                                 exists_recently = recent_check_cache.get(url)
            #                                 if exists_recently is None:
            #                                     exists_recently = await embedding.check_if_embedded_recently(filter={"source_url": url}, days=2)
            #                                     recent_check_cache[url] = exists_recently
            #                                 should_process = not exists_recently
            #                                 if not should_process:
            #                                     logging.info(f"      Skipping undated resource (found in recent DB): {url}")
            #
            #                             if should_process:
            #                                 if DEDUP_BY_URL:
            #                                     exists_in_db = db_url_exists_cache.get(url)
            #                                     if exists_in_db is None:
            #                                         exists_in_db = await embedding.url_exists_in_db(url)
            #                                         db_url_exists_cache[url] = exists_in_db
            #                                     if exists_in_db:
            #                                         logging.info(f"      Duplicate URL already in DB, skipping: {url}")
            #                                         continue
            #
            #                                 if url in seen_urls:
            #                                     logging.info(f"      Duplicate URL (this run), skipping: {url}")
            #                                 else:
            #                                     logging.info(f"      ADDING resource to process queue: {title} (Section: {section_tag})")
            #                                     resources_to_process.append((url, title, parsed_date, class_name, section_tag))
            #                                     seen_urls.add(url)
            #
            #                         except Exception as item_err:
            #                             logging.warning(f"      Could not process one item in {section_tag}: {item_err}")
            #                             continue
            #
            #                 except Exception as section_err:
            #                     logging.warning(f"   Skipping section '{section_tag}'. Could not find header or items. (Error: {section_err})")
            #                     continue
            #
            #     # Static syllabus scraping and embedding removed; topics inferred from general materials
            #
            #
            #     except Exception as course_err:
            #         logging.warning(f"Skipping course {class_name} due to critical navigation error: {course_err}")
            #         # Ensure we navigate back to a known state if possible
            #         try: await page.goto(config.COURSES_URL)
            #         except Exception: logging.error("Failed to navigate back to courses page after error.")
            #         continue # Skip to next course
            #
            # logging.info(f"\n--- Navigation complete. Found {len(resources_to_process)} candidate resources to process. ---")

            # --- TEMPORARY TEST BLOCK FOR ONE ZOOM LINK ---
            logging.info("\n--- SKIPPING NAVIGATION - RUNNING TARGETED ZOOM TEST ---")

            # This is one of the URLs that failed in your log
            test_url = "https://us06web.zoom.us/rec/share/rSRW_E-Cdk0hz9VmOEDW16_HVBzKfOw6Z6AGZwPQB8yh0UNWMQeSYQQBPLT7eyUW.EMUbqskKRju5waUL"
            test_title = "How does a machine learn to speak our language?"
            test_class = "AIML" # Must match your persona.json
            test_section = "sessions"

            # Add just this one item to the processing queue
            resources_to_process.append((test_url, test_title, None, test_class, test_section))

            logging.info(f"Added 1 test item to the queue: {test_title}")
            # --- END TEMPORARY TEST BLOCK ---

            # --- Processing Phase: Handle collected resource links ---
            logging.info("\n--- Starting Processing Phase ---")
            for url, title, date_obj, class_name, section_tag in resources_to_process:
                # Process each resource - this function handles type detection, download/scrape, refinery, embedding
                await process_single_resource(context, url, title, date_obj, class_name, section_tag)

    except Exception as pipeline_err:
        logging.critical(f"Pipeline failed with critical error: {pipeline_err}", exc_info=True)
        # Ensure browser is closed in case of crash
    finally:
        if page: await page.close()
        if context: await context.close()
        if browser: await browser.close()
        logging.info("🚀 Hybrid Pipeline Finished.")

# --- Allow running the script directly ---
if __name__ == "__main__":
    # --- Argument Parsing for Mode (Optional) ---
    # import argparse
    # parser = argparse.ArgumentParser(description="Run the Harvester & Refinery Pipeline.")
    # parser.add_argument('--mode', type=str, default="daily", choices=['daily', 'backlog'], help="Pipeline mode: 'daily' or 'backlog'.")
    # args = parser.parse_args()
    # asyncio.run(main_pipeline(mode=args.mode))

    # --- Default Run (e.g., for GitHub Actions) ---
    # Reads mode from environment variable or defaults to 'daily'
    pipeline_mode = os.environ.get("PIPELINE_MODE", "daily").lower()
    if pipeline_mode not in ["daily", "backlog"]:
        logging.warning(f"Invalid PIPELINE_MODE '{pipeline_mode}'. Defaulting to 'daily'.")
        pipeline_mode = "daily"

    asyncio.run(main_pipeline(mode=pipeline_mode))