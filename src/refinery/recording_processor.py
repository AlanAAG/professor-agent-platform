import logging
import os
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium import webdriver

# Import harvester-side config and navigation helpers via absolute package path
from src.harvester import config
from src.harvester.navigation import safe_find, safe_find_all, safe_click


def scrape_drive_transcript_content(driver: webdriver.Chrome) -> str:
    """Scrape transcript content from a Google Drive recording preview page.

    Retained for future use. Not invoked by extract_transcript when resource_type
    is DRIVE_RECORDING (we skip that flow by design to save resources).
    """
    logging.info("   Attempting Google Drive transcript scrape...")
    raw_transcription = ""
    try:
        # Use settings for wait timeout
        wait_timeout = getattr(config.SETTINGS, "wait_timeout", 30)
        wait = WebDriverWait(driver, wait_timeout)

        # Try to click play
        try:
            play_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_PLAY_BUTTON_CSS)))
        except TimeoutException:
            logging.info(
                f"      Play button not found within timeout (selector: {config.DRIVE_PLAY_BUTTON_CSS}); proceeding..."
            )
            play_btn = None
        except (NoSuchElementException, StaleElementReferenceException):
            logging.info("      Play button lookup failed; proceeding anyway.")
            play_btn = None
        if play_btn is not None:
            try:
                driver.execute_script("arguments[0].click();", play_btn)
                try:
                    # Wait for settings gear to indicate player controls are up
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_SETTINGS_BUTTON_CSS)))
                except TimeoutException:
                    pass
            except WebDriverException as e:
                logging.debug(f"      Ignoring error clicking play: {e}")

        # Open settings gear
        try:
            safe_click(driver, (By.CSS_SELECTOR, config.DRIVE_SETTINGS_BUTTON_CSS), timeout=wait_timeout)
        except TimeoutException:
            logging.error(
                f"      Settings button not clickable (selector: {config.DRIVE_SETTINGS_BUTTON_CSS})."
            )
            raise
        except NoSuchElementException:
            logging.error(
                f"      Settings button not present (selector: {config.DRIVE_SETTINGS_BUTTON_CSS})."
            )
            raise

        # Wait for transcript UI to appear
        try:
            wait.until(
                EC.any_of(
                    EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_HEADING_CSS)),
                    EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_CONTAINER_CSS)),
                )
            )
        except TimeoutException:
            pass

        # Container and segments
        try:
            container = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_CONTAINER_CSS))
            )
        except TimeoutException:
            logging.error(
                f"      Transcript container not found (selector: {config.DRIVE_TRANSCRIPT_CONTAINER_CSS})."
            )
            # Capture a screenshot right where it fails
            try:
                os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
                driver.save_screenshot(
                    os.path.join(
                        config.SETTINGS.screenshot_dir,
                        f"drive_no_container_{int(time.time())}.png",
                    )
                )
            except (OSError, WebDriverException):
                pass
            raise

        # Scroll to bottom repeatedly to load all content
        last_height = driver.execute_script("return arguments[0].scrollHeight", container)
        scroll_attempts = 0
        while scroll_attempts < 60:  # Max 60 attempts (3 minutes)
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", container)
            time.sleep(0.5)  # Wait half a second for content to appear
            new_height = driver.execute_script("return arguments[0].scrollHeight", container)
            if new_height == last_height:
                break  # Scrolled to bottom, no more content loading
            last_height = new_height
            scroll_attempts += 1

        segments = driver.find_elements(By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_SEGMENT_CSS)
        texts = [seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()]

        # Join segments with a space to preserve continuity
        raw_transcription = " ".join(texts)

        if raw_transcription:
            logging.info(
                f"   Successfully scraped Google Drive transcript ({len(raw_transcription)} chars)."
            )
        else:
            logging.warning("   Google Drive transcript segments found but were empty.")

    except TimeoutException as e:
        logging.error(f"   Timed out during Google Drive transcript scraping: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(
                os.path.join(
                    config.SETTINGS.screenshot_dir,
                    f"drive_scrape_timeout_{int(time.time())}.png",
                )
            )
        except (OSError, WebDriverException):
            pass
    except NoSuchElementException as e:
        logging.error(f"   Missing expected element on Google Drive page: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(
                os.path.join(
                    config.SETTINGS.screenshot_dir,
                    f"drive_scrape_no_such_{int(time.time())}.png",
                )
            )
        except (OSError, WebDriverException):
            pass
    except WebDriverException as e:
        logging.error(f"   Failed during Google Drive transcript scraping: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(
                os.path.join(
                    config.SETTINGS.screenshot_dir,
                    f"drive_scrape_error_{int(time.time())}.png",
                )
            )
        except (OSError, WebDriverException):
            pass

    return raw_transcription


def scrape_zoom_transcript_content(driver: webdriver.Chrome) -> str:
    """Scrape transcript content from a Zoom cloud recording page."""
    logging.info("   Attempting Zoom transcript scrape...")
    raw_transcription = ""
    wait_timeout = getattr(config.SETTINGS, "wait_timeout", 30)
    wait = WebDriverWait(driver, wait_timeout)

    try:
        # Best-effort interaction to initialize player/transcript visibility
        for sel in config.ZOOM_INITIAL_INTERACTIONS:
            try:
                driver.execute_script("document.querySelector(arguments[0])?.click();", sel)
                time.sleep(0.5)
            except WebDriverException:
                pass

        # Wait for transcript list and texts
        safe_find(driver, (By.CSS_SELECTOR, config.ZOOM_TRANSCRIPT_LIST_CSS), timeout=wait_timeout)
        safe_find(driver, (By.CSS_SELECTOR, config.ZOOM_TRANSCRIPT_TEXT_CSS), timeout=wait_timeout)

        texts = [el.text for el in safe_find_all(driver, (By.CSS_SELECTOR, config.ZOOM_TRANSCRIPT_TEXT_CSS), timeout=10) or []]

        # Filter and join text segments
        raw_transcription = " ".join([t.strip() for t in texts if t and t.strip()])

        if raw_transcription:
            logging.info(f"   Successfully scraped Zoom transcript ({len(raw_transcription)} chars).")
        else:
            logging.warning("   Zoom transcript items found but were empty.")

    except TimeoutException as e:
        logging.error(f"   Timed out while scraping Zoom transcript: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(
                os.path.join(
                    config.SETTINGS.screenshot_dir,
                    f"zoom_scrape_timeout_{int(time.time())}.png",
                )
            )
        except (OSError, WebDriverException):
            pass
    except NoSuchElementException as e:
        logging.error(f"   Missing expected element on Zoom page: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(
                os.path.join(
                    config.SETTINGS.screenshot_dir,
                    f"zoom_scrape_no_such_{int(time.time())}.png",
                )
            )
        except (OSError, WebDriverException):
            pass
    except WebDriverException as e:
        logging.error(f"   Failed to scrape Zoom transcript: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(
                os.path.join(
                    config.SETTINGS.screenshot_dir,
                    f"zoom_scrape_error_{int(time.time())}.png",
                )
            )
        except (OSError, WebDriverException):
            pass

    return raw_transcription


def extract_transcript(driver: webdriver.Chrome, url: str, resource_type: str) -> str:
    """Public entry point to selectively extract transcript content from a recording URL.

    - For ZOOM_RECORDING: open in a new tab and scrape transcript content.
    - For DRIVE_RECORDING: open in a new tab and scrape the Google Drive transcript.
    """
    original = driver.current_window_handle
    new_handle = None

    try:
        logging.info(f"   Opening transcript URL in new tab: {url}")

        # Open in new tab to preserve course page
        driver.execute_script("window.open(arguments[0], '_blank');", url)

        # Wait for new window handle to appear
        WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
            lambda d: len(d.window_handles) > 1
        )

        # Switch to the new tab
        new_handle = [h for h in driver.window_handles if h != original][0]
        driver.switch_to.window(new_handle)

        # Wait for page to begin loading content and check platform
        try:
            WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
                EC.any_of(
                    EC.url_contains("drive.google.com"),
                    EC.url_contains("zoom.us"),
                )
            )
        except WebDriverException:
            pass

        # Selective execution
        if resource_type == "ZOOM_RECORDING":
            return scrape_zoom_transcript_content(driver)
        if resource_type == "DRIVE_RECORDING":
            return scrape_drive_transcript_content(driver)

        # Unknown/unsupported types: do nothing
        logging.warning(f"   Unknown recording type: {resource_type}. Skipping.")
        return ""

    except TimeoutException as e:
        logging.error(f"   Timeout when opening or processing transcript URL {url}: {e}")
        return ""
    except WebDriverException as e:
        logging.error(f"   Error navigating to or processing transcript URL {url}: {e}")
        return ""
    finally:
        # Always close the new tab and switch back to the original
        if new_handle:
            try:
                driver.close()
            except Exception:
                pass  # Already closed/stale
        try:
            driver.switch_to.window(original)
        except Exception as e:
            logging.error(f"Failed to switch back to original window: {e}")
