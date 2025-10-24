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
    """
    Robustly scrapes Zoom transcript HTML and returns a normalized text string
    with lines formatted as: "[MM:SS] Speaker: text" (when available).

    Handles lazy loading by scrolling the transcript container to the top and
    triggering the "Resume autoscrolling" state if present.
    """
    logging.info("   Attempting Zoom transcript scrape...")
    raw_transcription = ""
    try:
        # Ensure the transcript container exists
        container = page.locator(config.ZOOM_TRANSCRIPT_CONTAINER_SELECTOR).first
        await container.wait_for(state="visible", timeout=30000)

        # If Zoom shows a "Resume autoscrolling" button, click it to load items
        try:
            resume_btn = page.locator(config.ZOOM_TRANSCRIPT_RESUME_AUTOSCROLL_BUTTON).first
            if await resume_btn.count() > 0 and await resume_btn.is_visible():
                logging.info("      Clicking 'Resume autoscrolling' to load full transcript...")
                await resume_btn.click()
                await page.wait_for_timeout(800)
        except Exception:
            pass

        # Scroll transcript viewport to top to ensure items load
        try:
            scroll_wrap = page.locator(config.ZOOM_TRANSCRIPT_SCROLL_WRAPPER_SELECTOR).first
            if await scroll_wrap.count() > 0:
                await scroll_wrap.evaluate("el => el.scrollTop = 0")
                await page.wait_for_timeout(500)
        except Exception:
            pass

        # Wait for the transcript list items to be present
        await page.locator(config.ZOOM_TRANSCRIPT_ITEM_SELECTOR).first.wait_for(
            state="visible", timeout=30000
        )

        # Try to load all items by scrolling to bottom until count stabilizes
        items = page.locator(config.ZOOM_TRANSCRIPT_ITEM_SELECTOR)
        last_count = -1
        stable_rounds = 0
        for _ in range(20):  # safety cap
            count = await items.count()
            if count == last_count:
                stable_rounds += 1
            else:
                stable_rounds = 0
            if stable_rounds >= 2:
                break
            last_count = count
            # scroll to bottom to force lazy load
            try:
                scroll_wrap = page.locator(config.ZOOM_TRANSCRIPT_SCROLL_WRAPPER_SELECTOR).first
                if await scroll_wrap.count() > 0:
                    await scroll_wrap.evaluate("el => el.scrollTop = el.scrollHeight")
                    await page.wait_for_timeout(400)
                else:
                    # fallback: scroll the container element
                    await container.evaluate("el => el.scrollTop = el.scrollHeight")
                    await page.wait_for_timeout(400)
            except Exception:
                await page.mouse.wheel(0, 2000)
                await page.wait_for_timeout(400)

        count = await items.count()
        logging.info(f"      Collected approximately {count} Zoom transcript list items after scrolling.")
        
        # Extract each item atomically to avoid large textContent concatenation issues
        segments: list[str] = []
        for i in range(count):
            item = items.nth(i)
            # Speaker name
            speaker = await item.locator(config.ZOOM_TRANSCRIPT_SPEAKER_SELECTOR).first.text_content() or ""
            speaker = speaker.strip()

            # Timestamp (if present)
            try:
                ts = await item.locator(config.ZOOM_TRANSCRIPT_TIME_SELECTOR).first.text_content()
            except Exception:
                ts = ""
            ts = (ts or "").strip()

            # Text content (main utterance)
            try:
                text = await item.locator(config.ZOOM_TRANSCRIPT_TEXT_SELECTOR).first.text_content()
            except Exception:
                text = ""
            text = (text or "").strip()

            if not text:
                continue

            # Build normalized line
            if ts and speaker:
                segments.append(f"[{ts}] {speaker}: {text}")
            elif ts:
                segments.append(f"[{ts}] {text}")
            elif speaker:
                segments.append(f"{speaker}: {text}")
            else:
                segments.append(text)

        raw_transcription = "\n".join(segments).strip()

        if raw_transcription:
            logging.info(f"   Successfully scraped Zoom transcript ({len(raw_transcription)} chars, {len(segments)} segments).")
        else:
            logging.warning("   Zoom transcript items found but produced empty text.")

    except Exception as e:
        logging.error(f"   Failed to scrape Zoom transcript: {e}", exc_info=True)
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

    