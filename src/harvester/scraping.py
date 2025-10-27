# src/harvester/scraping.py

import logging
import time
import os
import random
from typing import Optional

import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium import webdriver
from . import config
from .navigation import safe_find, safe_find_all, safe_click

# Use a browser-like User-Agent to avoid 403s from some sites
BROWSER_HEADER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

def _scrape_drive_transcript(driver: webdriver.Chrome) -> str:
    logging.info("   Attempting Google Drive transcript scrape...")
    raw_transcription = ""
    try:
        wait = WebDriverWait(driver, getattr(config, "SETTINGS", None).wait_timeout if hasattr(config, "SETTINGS") else 30)
        # Try to click play
        try:
            play_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_PLAY_BUTTON_CSS)))
        except TimeoutException:
            logging.info(f"      Play button not found within timeout (selector: {config.DRIVE_PLAY_BUTTON_CSS}); proceeding...")
            play_btn = None
        except (NoSuchElementException, StaleElementReferenceException):
            logging.info("      Play button lookup failed; proceeding anyway.")
            play_btn = None
        if play_btn is not None:
            try:
                driver.execute_script("arguments[0].click();", play_btn)
                try:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_SETTINGS_BUTTON_CSS)))
                except TimeoutException:
                    pass
            except WebDriverException as e:
                logging.debug(f"      Ignoring error clicking play: {e}")

        # Open settings gear
        try:
            safe_click(driver, (By.CSS_SELECTOR, config.DRIVE_SETTINGS_BUTTON_CSS))
        except TimeoutException:
            logging.error(f"      Settings button not clickable (selector: {config.DRIVE_SETTINGS_BUTTON_CSS}).")
            raise
        except NoSuchElementException:
            logging.error(f"      Settings button not present (selector: {config.DRIVE_SETTINGS_BUTTON_CSS}).")
            raise
        # Wait for transcript UI to appear
        try:
            wait.until(EC.any_of(
                EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_HEADING_CSS)),
                EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_CONTAINER_CSS)),
            ))
        except TimeoutException:
            pass

        # Wait for transcript heading or container
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_HEADING_CSS)))
        except TimeoutException:
            logging.info("      Transcript heading not found; checking container directly...")

        # Container and segments
        try:
            container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_CONTAINER_CSS)))
        except TimeoutException:
            logging.error(f"      Transcript container not found (selector: {config.DRIVE_TRANSCRIPT_CONTAINER_CSS}).")
            # Capture a screenshot right where it fails
            try:
                os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
                driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"drive_no_container_{int(time.time())}.png"))
            except (OSError, WebDriverException):
                pass
            raise
        # Scroll to bottom to load all with explicit wait for growth
        last_height = driver.execute_script("return arguments[0].scrollHeight", container)
        for _ in range(60):
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", container)
            try:
                WebDriverWait(driver, 3).until(
                    lambda d: d.execute_script("return arguments[0].scrollHeight", container) > last_height
                )
                last_height = driver.execute_script("return arguments[0].scrollHeight", container)
            except TimeoutException:
                break

        segments = driver.find_elements(By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_SEGMENT_CSS)
        texts = [seg.text.strip() for seg in segments if getattr(seg, 'text', '').strip()]
        raw_transcription = " ".join(texts)
        if raw_transcription:
            logging.info(f"   Successfully scraped Google Drive transcript ({len(raw_transcription)} chars).")
        else:
            logging.warning("   Google Drive transcript segments found but were empty.")
    except TimeoutException as e:
        logging.error(f"   Timed out during Google Drive transcript scraping: {e}")
        try:
            os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
            driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"drive_scrape_timeout_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    except NoSuchElementException as e:
        logging.error(f"   Missing expected element on Google Drive page: {e}")
        try:
            os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
            driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"drive_scrape_no_such_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    except WebDriverException as e:
        logging.error(f"   Failed during Google Drive transcript scraping: {e}")
        try:
            os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
            driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"drive_scrape_error_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    return raw_transcription

def _scrape_zoom_transcript(driver: webdriver.Chrome) -> str:
    logging.info("   Attempting Zoom transcript scrape...")
    raw_transcription = ""
    wait = WebDriverWait(driver, getattr(config, "SETTINGS", None).wait_timeout if hasattr(config, "SETTINGS") else 30)
    try:
        # Cookie buttons (best-effort)
        for label in ["Accept", "Agree", "Got it", "Allow", "Cookies Settings"]:
            try:
                btn = wait.until(EC.presence_of_element_located((By.XPATH, f"//button[contains(., '{label}')]")))
            except TimeoutException:
                continue
            try:
                btn.click()
                try:
                    WebDriverWait(driver, 2).until(EC.staleness_of(btn))
                except TimeoutException:
                    pass
                break
            except WebDriverException:
                # Non-fatal; continue with scraping
                pass

        # Light human-like scroll
        driver.execute_script("window.scrollBy(0, 200);")
        driver.execute_script("window.scrollTo(0, 0);")

        # Initialize player (best-effort)
        try:
            player = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.playback-video")))
        except TimeoutException:
            player = None
        except NoSuchElementException:
            player = None
        if player is not None:
            try:
                driver.execute_script("arguments[0].click();", player)
            except WebDriverException:
                pass
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.ZOOM_TRANSCRIPT_LIST_CSS)))
            except TimeoutException:
                pass

        # Wait for transcript list and texts
        try:
            safe_find(driver, (By.CSS_SELECTOR, config.ZOOM_TRANSCRIPT_LIST_CSS), timeout=30)
        except TimeoutException:
            logging.error(f"      Transcript list not found (selector: {config.ZOOM_TRANSCRIPT_LIST_CSS}).")
            try:
                os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
                driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"zoom_no_list_{int(time.time())}.png"))
            except (OSError, WebDriverException):
                pass
            raise
        try:
            safe_find(driver, (By.CSS_SELECTOR, config.ZOOM_TRANSCRIPT_TEXT_CSS), timeout=20)
        except TimeoutException:
            logging.error(f"      Transcript text not visible (selector: {config.ZOOM_TRANSCRIPT_TEXT_CSS}).")
            try:
                os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
                driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"zoom_text_not_visible_{int(time.time())}.png"))
            except (OSError, WebDriverException):
                pass
            raise
        texts = [el.text for el in safe_find_all(driver, (By.CSS_SELECTOR, config.ZOOM_TRANSCRIPT_TEXT_CSS), timeout=10) or []]
        raw_transcription = " ".join([t.strip() for t in texts if t and t.strip()])
        if raw_transcription:
            logging.info(f"   Successfully scraped Zoom transcript ({len(raw_transcription)} chars).")
        else:
            logging.warning("   Zoom transcript items found but were empty.")
    except TimeoutException as e:
        logging.error(f"   Timed out while scraping Zoom transcript: {e}")
        try:
            os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
            driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"zoom_scrape_timeout_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    except NoSuchElementException as e:
        logging.error(f"   Missing expected element on Zoom page: {e}")
        try:
            os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
            driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"zoom_scrape_no_such_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    except WebDriverException as e:
        logging.error(f"   Failed to scrape Zoom transcript: {e}")
        try:
            os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
            driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"zoom_scrape_error_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    return raw_transcription

def scrape_transcript_from_url(driver: webdriver.Chrome, url: str) -> str:
    try:
        logging.info(f"   Opening transcript URL: {url}")
        # Open in new tab to preserve course page
        original = driver.current_window_handle
        driver.execute_script("window.open(arguments[0], '_blank');", url)
        WebDriverWait(driver, getattr(config, "SETTINGS", None).wait_timeout if hasattr(config, "SETTINGS") else 30).until(lambda d: len(d.window_handles) > 1)
        new_handle = [h for h in driver.window_handles if h != original][0]
        driver.switch_to.window(new_handle)
        # Wait for page to begin loading content
        try:
            WebDriverWait(driver, getattr(config, "SETTINGS", None).wait_timeout if hasattr(config, "SETTINGS") else 20).until(
                EC.any_of(
                    EC.url_contains("drive.google.com"),
                    EC.url_contains("zoom.us"),
                )
            )
        except WebDriverException:
            pass
        current_url = driver.current_url
        if "drive.google.com" in current_url:
            text = _scrape_drive_transcript(driver)
        elif "zoom.us" in current_url:
            text = _scrape_zoom_transcript(driver)
        else:
            logging.warning(f"   Unknown recording platform at {current_url}. Skipping transcript scrape.")
            text = ""
        driver.close()
        driver.switch_to.window(original)
        return text
    except TimeoutException as e:
        logging.error(f"   Timeout when opening or processing transcript URL {url}: {e}")
        try:
            os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
            driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"nav_timeout_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
        try:
            driver.switch_to.window(original)
        except WebDriverException:
            pass
        return ""
    except WebDriverException as e:
        logging.error(f"   Error navigating to or processing transcript URL {url}: {e}")
        try:
            os.makedirs(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), exist_ok=True)
            driver.save_screenshot(os.path.join(getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots"), f"nav_error_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
        try:
            # Ensure we return to original handle if possible
            for h in driver.window_handles[:-1]:
                pass
            if driver.window_handles:
                driver.switch_to.window(driver.window_handles[0])
        except WebDriverException:
            pass
        return ""


# --- NEW HELPER FUNCTIONS ---

def check_url_content_type(url: str) -> str:
    logging.debug(f"Checking content type for URL: {url}")
    if not url or not url.startswith(("http://", "https://")):
        logging.warning(f"Invalid URL for content type check: {url}")
        return "unknown"
    try:
        resp = requests.head(url, timeout=15, allow_redirects=True, headers=BROWSER_HEADER)
        resp.raise_for_status()
        return (resp.headers.get("Content-Type", "")).lower()
    except RequestException as e:
        logging.error(f"   Error checking content type for {url}: {e}")
        return "error"

def download_file(url: str, save_dir: str, filename: str) -> Optional[str]:
    filepath = os.path.join(save_dir, filename)
    logging.info(f"   Attempting to download file from {url} to {filepath}")
    try:
        with requests.get(url, timeout=60, allow_redirects=True, headers=BROWSER_HEADER, stream=True) as r:
            r.raise_for_status()
            os.makedirs(save_dir, exist_ok=True)
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f"   Successfully downloaded {filename}")
        return filepath
    except RequestException as e:
        logging.error(f"   Error downloading {url}: {e}")
        return None

def scrape_html_content(url: str) -> Optional[str]:
    logging.info(f"   Attempting to scrape HTML content from: {url}")
    try:
        resp = requests.get(url, timeout=30, allow_redirects=True, headers=BROWSER_HEADER)
        resp.raise_for_status()
        html_content = resp.text
        doc = ReadabilityDocument(html_content)
        main_content_html = doc.summary()
        soup = BeautifulSoup(main_content_html, 'lxml')
        main_text = soup.get_text(separator='\n', strip=True)
        logging.info(f"   Successfully scraped HTML content ({len(main_text)} chars).")
        return main_text
    except RequestException as e:
        logging.error(f"   Error scraping HTML {url}: {e}")
        return None

    