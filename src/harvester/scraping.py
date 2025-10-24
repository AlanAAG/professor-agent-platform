# src/harvester/scraping.py

import logging
import time
import asyncio
import os # For path joining in download
import aiohttp # For async HTTP requests
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument # To extract main content
from playwright.async_api import Page, BrowserContext
from . import config

# Use a browser-like User-Agent to avoid 403s from some sites
BROWSER_HEADER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# --- Existing Transcript Scraping Functions ---
async def _scrape_drive_transcript(page: Page) -> str:
    # ... (Keep existing implementation with Play/Settings/Transcript clicks) ...
    logging.info("   Attempting Google Drive transcript scrape...")
    raw_transcription = ""
    try:
        play_button = page.locator(config.DRIVE_VIDEO_PLAY_BUTTON).first
        # Ensure we wait for the element before checking visibility; Playwright's is_visible doesn't take timeout
        try:
            await play_button.wait_for(state="visible", timeout=10000)
            logging.info("      Clicking play button...")
            await play_button.click(timeout=5000)
            await page.wait_for_timeout(2000)
        except Exception:
            logging.info("      Play button not immediately visible or already played, proceeding...")

        settings_button = page.locator(config.DRIVE_SETTINGS_BUTTON).first
        logging.info("      Clicking settings (gear) button...")
        await settings_button.wait_for(state="visible", timeout=15000)
        await settings_button.click(timeout=5000)

        transcript_menu_item = page.locator(config.DRIVE_TRANSCRIPT_MENU_ITEM).first
        logging.info("      Clicking transcript menu item...")
        await transcript_menu_item.wait_for(state="visible", timeout=10000)
        await transcript_menu_item.click(timeout=5000)

        logging.info("      Waiting for transcript segments to appear...")
        transcript_segment_selector = config.DRIVE_TRANSCRIPT_SEGMENT_SELECTOR
        await page.wait_for_selector(transcript_segment_selector, state="visible", timeout=45000)

        all_segments = await page.locator(transcript_segment_selector).all_text_contents()
        raw_transcription = " ".join(filter(None, all_segments))

        if raw_transcription:
            logging.info(f"   Successfully scraped Google Drive transcript ({len(raw_transcription)} chars).")
        else:
            logging.warning("   Google Drive transcript segments found but were empty.")

    except Exception as e:
        logging.error(f"   Failed during Google Drive transcript scraping steps: {e}", exc_info=True)
        try: await page.screenshot(path=f"logs/error_screenshots/drive_scrape_error_{int(time.time())}.png")
        except Exception as screen_err: logging.error(f"      Failed to save error screenshot: {screen_err}")
    return raw_transcription

async def _scrape_zoom_transcript(page: Page) -> str:
    """Scrapes transcript text from a Zoom recording page."""
    logging.info("   Attempting Zoom transcript scrape...")
    raw_transcription = ""
    try:
        # Wait for the *list* of transcript items to be attached to the DOM
        list_selector = "ul.transcript-list"
        await page.wait_for_selector(list_selector, state="attached", timeout=30000)
        logging.info("      Transcript list container found.")

        # Target the specific text elements within the timeline
        text_selector = "div.timeline div.text"
        # Wait for the first text element to be visible
        await page.locator(text_selector).first.wait_for(state="visible", timeout=10000) 

        all_segments = await page.locator(text_selector).all_text_contents()
        raw_transcription = " ".join(filter(None, all_segments))

        if raw_transcription: 
            logging.info(f"   Successfully scraped Zoom transcript ({len(raw_transcription)} chars).")
        else: 
            logging.warning("   Zoom transcript items found but were empty.")
    except Exception as e:
        logging.error(f"   Failed to scrape Zoom transcript: {e}", exc_info=True)
        try: 
            await page.screenshot(path=f"logs/error_screenshots/zoom_scrape_error_{int(time.time())}.png")
        except Exception as screen_err: 
            logging.error(f"      Failed to save error screenshot: {screen_err}")
    return raw_transcription

async def scrape_transcript_from_url(context: BrowserContext, url: str) -> str:
    # ... (Keep existing implementation) ...
    page = None
    try:
        page = await context.new_page()
        logging.info(f"   Opening transcript URL: {url}")
        await page.goto(url, wait_until="load", timeout=90000)
        logging.info("      Waiting for page elements to potentially initialize...")
        await page.wait_for_timeout(7000)
        current_url = page.url
        if "drive.google.com" in current_url:
            return await _scrape_drive_transcript(page)
        elif "zoom.us" in current_url:
            return await _scrape_zoom_transcript(page)
        else:
            logging.warning(f"   Unknown recording platform at {current_url}. Skipping transcript scrape.")
            return ""
    except Exception as e:
        logging.error(f"   Error navigating to or processing transcript URL {url}: {e}", exc_info=True)
        if page:
            try: await page.screenshot(path=f"logs/error_screenshots/nav_error_{int(time.time())}.png")
            except Exception as screen_err: logging.error(f"      Failed to save navigation error screenshot: {screen_err}")
        return ""
    finally:
        if page:
            try: await page.close(); logging.info(f"   Closed tab for {url}")
            except Exception as close_err: logging.error(f"   Error closing page for {url}: {close_err}")


# --- NEW HELPER FUNCTIONS ---

async def check_url_content_type(url: str) -> str:
    """Sends a HEAD request to determine the content type (PDF, HTML, etc.)."""
    logging.debug(f"Checking content type for URL: {url}")
    # Handle potential Playwright internal URLs if passed accidentally
    if not url or not url.startswith(('http://', 'https://')):
        logging.warning(f"Invalid or internal URL passed to check_url_content_type: {url}")
        return "unknown"
    try:
        async with aiohttp.ClientSession() as session:
            # Use HEAD request to avoid downloading the whole file
            async with session.head(url, timeout=15, allow_redirects=True, headers=BROWSER_HEADER) as response:
                response.raise_for_status() # Raise error for bad status (4xx, 5xx)
                content_type = response.headers.get('Content-Type', '').lower()
                logging.debug(f"   URL: {url}, Content-Type: {content_type}")
                return content_type
    except aiohttp.ClientError as e:
        logging.error(f"   Network error checking content type for {url}: {e}")
        return "error"
    except asyncio.TimeoutError:
        logging.error(f"   Timeout checking content type for {url}")
        return "error"
    except Exception as e:
        logging.error(f"   Unexpected error checking content type for {url}: {e}")
        return "error"

async def download_file(url: str, save_dir: str, filename: str) -> str | None:
    """Downloads a file (e.g., PDF) from a URL asynchronously."""
    filepath = os.path.join(save_dir, filename)
    logging.info(f"   Attempting to download file from {url} to {filepath}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=60, allow_redirects=True, headers=BROWSER_HEADER) as response: # Longer timeout for downloads
                response.raise_for_status()
                # Ensure save directory exists
                os.makedirs(save_dir, exist_ok=True)
                # Stream the download
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192) # Read in chunks
                        if not chunk:
                            break
                        f.write(chunk)
                logging.info(f"   Successfully downloaded {filename}")
                return filepath
    except aiohttp.ClientError as e:
        logging.error(f"   Network error downloading {url}: {e}")
        return None
    except asyncio.TimeoutError:
        logging.error(f"   Timeout downloading {url}")
        return None
    except Exception as e:
        logging.error(f"   Unexpected error downloading {url}: {e}")
        return None

async def scrape_html_content(url: str) -> str | None:
    """Fetches an HTML page and extracts the main article text."""
    logging.info(f"   Attempting to scrape HTML content from: {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30, allow_redirects=True, headers=BROWSER_HEADER) as response:
                response.raise_for_status()
                html_content = await response.text()

                # Use readability-lxml to extract the main content
                doc = ReadabilityDocument(html_content)
                main_content_html = doc.summary() # Gets main content HTML
                
                # Convert main content HTML to clean text using BeautifulSoup
                soup = BeautifulSoup(main_content_html, 'lxml')
                main_text = soup.get_text(separator='\n', strip=True) # Get text, preserving paragraphs

                logging.info(f"   Successfully scraped HTML content ({len(main_text)} chars).")
                return main_text
    except aiohttp.ClientError as e:
        logging.error(f"   Network error scraping HTML {url}: {e}")
        return None
    except asyncio.TimeoutError:
        logging.error(f"   Timeout scraping HTML {url}")
        return None
    except Exception as e:
        logging.error(f"   Unexpected error scraping HTML {url}: {e}")
        return None

    