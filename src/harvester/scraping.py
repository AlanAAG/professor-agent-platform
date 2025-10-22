# src/harvester/scraping.py

import logging
import time # Added for potential short delays
from playwright.async_api import Page, BrowserContext # Added Page for type hint
from . import config # Imports selectors from config.py

async def _scrape_drive_transcript(page: Page) -> str: # Added Page type hint
    """
    Clicks through necessary buttons and scrapes transcript text from a Google Drive page.
    """
    logging.info("   Attempting Google Drive transcript scrape...")
    raw_transcription = ""
    try:
        # --- ADDED STEPS START ---
        
        # 1. Click Play Button (if necessary - check if video auto-plays)
        # It's safer to assume we need to click play.
        play_button = page.locator(config.DRIVE_VIDEO_PLAY_BUTTON).first
        if await play_button.is_visible(timeout=10000): # Wait up to 10s for play button
            logging.info("      Clicking play button...")
            await play_button.click(timeout=5000)
            await page.wait_for_timeout(2000) # Wait briefly for player controls to likely appear
        else:
            logging.info("      Play button not immediately visible or already played, proceeding...")

        # 2. Click Settings (Gear) Button
        settings_button = page.locator(config.DRIVE_SETTINGS_BUTTON).first
        logging.info("      Clicking settings (gear) button...")
        await settings_button.wait_for(state="visible", timeout=15000)
        await settings_button.click(timeout=5000)
        
        # 3. Click Transcript Menu Item
        # Wait for the menu item itself to become visible after clicking settings
        transcript_menu_item = page.locator(config.DRIVE_TRANSCRIPT_MENU_ITEM).first
        logging.info("      Clicking transcript menu item...")
        await transcript_menu_item.wait_for(state="visible", timeout=10000)
        await transcript_menu_item.click(timeout=5000)
        
        # --- ADDED STEPS END ---

        # 4. Wait for and Scrape Transcript Segments (Existing Logic)
        logging.info("      Waiting for transcript segments to appear...")
        transcript_segment_selector = config.DRIVE_TRANSCRIPT_SEGMENT_SELECTOR
        # Wait for the *first* segment to ensure the panel is loaded
        await page.wait_for_selector(transcript_segment_selector, state="visible", timeout=45000) # Keep increased timeout

        # Get all segments that match
        all_segments = await page.locator(transcript_segment_selector).all_text_contents()
        raw_transcription = " ".join(filter(None, all_segments))

        if raw_transcription:
            logging.info(f"   Successfully scraped Google Drive transcript ({len(raw_transcription)} chars).")
        else:
            logging.warning("   Google Drive transcript segments found but were empty.")

    except Exception as e:
        # Catch timeout or other errors during the multi-step process
        logging.error(f"   Failed during Google Drive transcript scraping steps: {e}", exc_info=True) # Log traceback
        try: # Try to save screenshot even if page reference might be tricky
            await page.screenshot(path=f"logs/error_screenshots/drive_scrape_error_{int(time.time())}.png")
        except Exception as screen_err:
            logging.error(f"      Failed to save error screenshot: {screen_err}")
    return raw_transcription # Return scraped text or empty string on failure

# --- (Keep _scrape_zoom_transcript and scrape_transcript_from_url as they were) ---
async def _scrape_zoom_transcript(page: Page) -> str:
    """Scrapes transcript text from a Zoom recording page."""
    logging.info("   Attempting Zoom transcript scrape...")
    transcript_container_selector = config.ZOOM_TRANSCRIPT_CONTAINER_SELECTOR
    raw_transcription = ""
    try:
        container = page.locator(transcript_container_selector).first
        await container.wait_for(state="visible", timeout=30000)
        raw_transcription = await container.text_content(timeout=5000)
        if raw_transcription:
            logging.info(f"   Successfully scraped Zoom transcript ({len(raw_transcription)} chars).")
        else:
            logging.warning("   Zoom transcript container found but was empty.")
    except Exception as e:
        logging.error(f"   Failed to scrape Zoom transcript: {e}", exc_info=True)
        # await page.screenshot(path=f"logs/error_screenshots/zoom_scrape_error_{int(time.time())}.png")
    return raw_transcription

async def scrape_transcript_from_url(context: BrowserContext, url: str) -> str:
    """Opens URL in new tab, detects platform, scrapes transcript, closes tab."""
    page = None
    try:
        page = await context.new_page()
        logging.info(f"   Opening URL: {url}")
        # Use 'load' or 'domcontentloaded', maybe increase timeout further if pages are slow
        await page.goto(url, wait_until="load", timeout=90000) # Increased timeout, wait for full load

        # Maybe wait longer for video players to initialize
        logging.info("      Waiting for page elements to potentially initialize...")
        await page.wait_for_timeout(7000) # Increased wait

        current_url = page.url # Get URL after potential redirects

        if "drive.google.com" in current_url:
            return await _scrape_drive_transcript(page) # Call updated function
        elif "zoom.us" in current_url:
            return await _scrape_zoom_transcript(page)
        else:
            logging.warning(f"   Unknown recording platform at {current_url}. Skipping scrape.")
            return ""
    except Exception as e:
        logging.error(f"   Error navigating to or processing URL {url}: {e}", exc_info=True)
        if page:
            try:
                await page.screenshot(path=f"logs/error_screenshots/nav_error_{int(time.time())}.png")
            except Exception as screen_err:
                logging.error(f"      Failed to save navigation error screenshot: {screen_err}")
        return ""
    finally:
        if page:
            try:
                await page.close()
                logging.info(f"   Closed tab for {url}")
            except Exception as close_err:
                logging.error(f"   Error closing page for {url}: {close_err}")