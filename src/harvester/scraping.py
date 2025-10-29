# src/harvester/scraping.py

import logging
import time
import os
import random
import re
from typing import Optional, List, Dict, Any

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
from selenium.webdriver.remote.webelement import WebElement
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
        # Use settings for wait timeout
        wait_timeout = getattr(config.SETTINGS, "wait_timeout", 30)
        wait = WebDriverWait(driver, wait_timeout)
        
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

        # Container and segments
        try:
            container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_CONTAINER_CSS)))
        except TimeoutException:
            logging.error(f"      Transcript container not found (selector: {config.DRIVE_TRANSCRIPT_CONTAINER_CSS}).")
            # Capture a screenshot right where it fails
            try:
                os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
                driver.save_screenshot(os.path.join(config.SETTINGS.screenshot_dir, f"drive_no_container_{int(time.time())}.png"))
            except (OSError, WebDriverException):
                pass
            raise
            
        # Scroll to bottom repeatedly to load all content
        last_height = driver.execute_script("return arguments[0].scrollHeight", container)
        scroll_attempts = 0
        while scroll_attempts < 60: # Max 60 attempts (3 minutes)
            driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", container)
            time.sleep(0.5) # Wait half a second for content to appear
            new_height = driver.execute_script("return arguments[0].scrollHeight", container)
            if new_height == last_height:
                break # Scrolled to bottom, no more content loading
            last_height = new_height
            scroll_attempts += 1

        segments = driver.find_elements(By.CSS_SELECTOR, config.DRIVE_TRANSCRIPT_SEGMENT_CSS)
        texts = [seg.text.strip() for seg in segments if getattr(seg, 'text', '').strip()]
        
        # Join segments with a space to preserve continuity
        raw_transcription = " ".join(texts)
        
        if raw_transcription:
            logging.info(f"   Successfully scraped Google Drive transcript ({len(raw_transcription)} chars).")
        else:
            logging.warning("   Google Drive transcript segments found but were empty.")
            
    except TimeoutException as e:
        logging.error(f"   Timed out during Google Drive transcript scraping: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(os.path.join(config.SETTINGS.screenshot_dir, f"drive_scrape_timeout_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    except NoSuchElementException as e:
        logging.error(f"   Missing expected element on Google Drive page: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(os.path.join(config.SETTINGS.screenshot_dir, f"drive_scrape_no_such_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    except WebDriverException as e:
        logging.error(f"   Failed during Google Drive transcript scraping: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(os.path.join(config.SETTINGS.screenshot_dir, f"drive_scrape_error_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
            
    return raw_transcription

def _scrape_zoom_transcript(driver: webdriver.Chrome) -> str:
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
            driver.save_screenshot(os.path.join(config.SETTINGS.screenshot_dir, f"zoom_scrape_timeout_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    except NoSuchElementException as e:
        logging.error(f"   Missing expected element on Zoom page: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(os.path.join(config.SETTINGS.screenshot_dir, f"zoom_scrape_no_such_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
    except WebDriverException as e:
        logging.error(f"   Failed to scrape Zoom transcript: {e}")
        try:
            os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
            driver.save_screenshot(os.path.join(config.SETTINGS.screenshot_dir, f"zoom_scrape_error_{int(time.time())}.png"))
        except (OSError, WebDriverException):
            pass
            
    return raw_transcription

def scrape_transcript_from_url(driver: webdriver.Chrome, url: str) -> str:
    """Navigates to a recording URL and attempts to scrape the transcript."""
    original = driver.current_window_handle
    new_handle = None
    
    try:
        logging.info(f"   Opening transcript URL in new tab: {url}")
        
        # Open in new tab to preserve course page
        driver.execute_script("window.open(arguments[0], '_blank');", url)
        
        # Wait for new window handle to appear
        WebDriverWait(driver, config.SETTINGS.wait_timeout).until(lambda d: len(d.window_handles) > 1)
        
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
            
        current_url = driver.current_url
        text = ""
        
        if "drive.google.com" in current_url:
            text = _scrape_drive_transcript(driver)
        elif "zoom.us" in current_url:
            text = _scrape_zoom_transcript(driver)
        else:
            logging.warning(f"   Unknown recording platform at {current_url}. Skipping transcript scrape.")
            
        return text
        
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
                pass # Already closed/stale
        try:
            driver.switch_to.window(original)
        except Exception as e:
            logging.error(f"Failed to switch back to original window: {e}")


# --- Static Resource Helpers (Uses requests) ---

def check_url_content_type(url: str) -> str:
    """Checks the Content-Type header of a URL using a HEAD request."""
    logging.debug(f"Checking content type for URL: {url}")
    if not url or not url.startswith(("http://", "https://")):
        logging.warning(f"Invalid URL for content type check: {url}")
        return "unknown"
    try:
        # Use HEAD request to check headers without downloading content
        resp = requests.head(url, timeout=15, allow_redirects=True, headers=BROWSER_HEADER)
        resp.raise_for_status()
        return (resp.headers.get("Content-Type", "")).lower()
    except RequestException as e:
        logging.error(f"   Error checking content type for {url}: {e}")
        return "error"

def download_file(url: str, save_dir: str, filename: str) -> Optional[str]:
    """Downloads a file (e.g., PDF) using a GET request."""
    filepath = os.path.join(save_dir, filename)
    logging.info(f"   Attempting to download file from {url} to {filepath}")
    try:
        with requests.get(url, timeout=60, allow_redirects=True, headers=BROWSER_HEADER, stream=True) as r:
            r.raise_for_status()
            os.makedirs(save_dir, exist_ok=True)
            with open(filepath, "wb") as f:
                # Stream the content in chunks
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logging.info(f"   Successfully downloaded {filename}")
        return filepath
    except RequestException as e:
        logging.error(f"   Error downloading {url}: {e}")
        return None

def scrape_html_content(url: str) -> Optional[str]:
    """Scrapes and extracts the main readable text from a general webpage."""
    logging.info(f"   Attempting to scrape HTML content from: {url}")
    try:
        # Use GET request for full content
        resp = requests.get(url, timeout=30, allow_redirects=True, headers=BROWSER_HEADER)
        resp.raise_for_status()
        html_content = resp.text
        
        # Use readability-lxml to extract the main article content
        doc = ReadabilityDocument(html_content)
        main_content_html = doc.summary()
        
        # Use BeautifulSoup to convert HTML summary to clean text
        soup = BeautifulSoup(main_content_html, 'lxml')
        main_text = soup.get_text(separator='\n', strip=True)
        
        logging.info(f"   Successfully scraped HTML content ({len(main_text)} chars).")
        return main_text
    except RequestException as e:
        logging.error(f"   Error scraping HTML {url}: {e}")
        return None


# --- Resource Link Processing ---

def classify_url(url: str) -> str:
    """Classify a URL into a canonical resource type.

    Types: YOUTUBE_VIDEO, PDF_DOCUMENT, OFFICE_DOCUMENT, RECORDING_LINK, WEB_ARTICLE
    """
    if not url:
        return "WEB_ARTICLE"
    lower = url.lower()
    if "youtube.com/watch" in lower or "youtu.be/" in lower:
        return "YOUTUBE_VIDEO"
    if lower.endswith(".pdf"):
        return "PDF_DOCUMENT"
    if "docs.google.com/presentation" in lower or lower.endswith((".pptx", ".ppt")):
        return "OFFICE_DOCUMENT"
    if "docs.google.com/document" in lower or lower.endswith((".doc", ".docx")):
        return "OFFICE_DOCUMENT"
    if "docs.google.com/spreadsheets" in lower or lower.endswith((".xls", ".xlsx")):
        return "OFFICE_DOCUMENT"
    if "drive.google.com/file" in lower or "zoom.us/rec" in lower or "zoom.us/" in lower:
        return "RECORDING_LINK"
    return "WEB_ARTICLE"


def process_resource_links(driver: webdriver.Chrome, links: List[WebElement]) -> List[Dict[str, Any]]:
    """Process a list of <a> WebElements into structured resource metadata.

    For each link, extract:
      - url: anchor href
      - title: best-effort text from nearby context (p/heading/label)
      - date: date-like text (e.g., 29/09/2025) from adjacent span/containers
      - type: classified by URL pattern

    The DOM varies between regular resource links and session recordings.
    This function uses robust ancestor/sibling traversal with sensible fallbacks.
    """

    def _normalize_text(text: str) -> str:
        return " ".join((text or "").split())

    date_regex = re.compile(r"\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b")

    def _safe_attr(el: WebElement, name: str) -> str:
        try:
            return el.get_attribute(name) or ""
        except Exception:
            return ""

    def _try_find(el: WebElement, by: str, selector: str) -> WebElement | None:
        try:
            return el.find_element(by, selector)
        except Exception:
            return None

    def _try_find_all(el: WebElement, by: str, selector: str) -> List[WebElement]:
        try:
            return el.find_elements(by, selector) or []
        except Exception:
            return []

    def _nearest_container(anchor: WebElement) -> WebElement:
        # Prefer logical content containers (li/div/section/article) around the link
        container = _try_find(anchor, By.XPATH, "./ancestor::*[self::li or self::div or self::section or self::article][1]")
        return container or anchor

    def _extract_title(anchor: WebElement, url: str) -> str:
        # 1) Anchor text if meaningful
        anchor_text = _normalize_text(getattr(anchor, "text", ""))
        if anchor_text and anchor_text.lower() != (url or "").lower() and len(anchor_text) > 3:
            return anchor_text

        # 2) aria-label/title attributes on the anchor
        for attr_name in ("aria-label", "title"):
            val = _normalize_text(_safe_attr(anchor, attr_name))
            if val:
                return val

        # 3) Look for nearby <p> siblings
        sib_p = _try_find(anchor, By.XPATH, "following-sibling::p[normalize-space()][1]") or _try_find(
            anchor, By.XPATH, "preceding-sibling::p[normalize-space()][1]"
        )
        if sib_p:
            sib_text = _normalize_text(getattr(sib_p, "text", ""))
            if sib_text:
                return sib_text

        container = _nearest_container(anchor)

        # 4) Titles commonly live in <p> within the container (Openai, Georgetown, Session PPT, ...)
        p_in_container = _try_find(container, By.XPATH, ".//p[normalize-space()][1]")
        if p_in_container:
            p_text = _normalize_text(getattr(p_in_container, "text", ""))
            if p_text:
                return p_text

        # 5) Headings near the link
        headings = _try_find_all(
            container,
            By.XPATH,
            ".//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 or self::h6][normalize-space()]",
        )
        if headings:
            h_text = _normalize_text(getattr(headings[0], "text", ""))
            if h_text:
                return h_text

        # 6) End-of-section boxes (e.g., .fileBoxend) can hold a final title
        filebox = _try_find(
            container,
            By.XPATH,
            ".//div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxend')][1]",
        )
        if filebox:
            fb_p = _try_find(filebox, By.XPATH, ".//p[normalize-space()][1]")
            if fb_p:
                fb_text = _normalize_text(getattr(fb_p, "text", ""))
                if fb_text:
                    return fb_text

        # 7) Fallback: parent text trimmed (can be noisy but better than empty)
        parent = _try_find(anchor, By.XPATH, "parent::*")
        parent_text = _normalize_text(getattr(parent, "text", "")) if parent else ""
        return parent_text or url or ""

    def _extract_date(anchor: WebElement) -> str | None:
        # Check spans near the anchor first
        for xpath in (
            "following-sibling::span[normalize-space()][1]",
            "preceding-sibling::span[normalize-space()][1]",
        ):
            span = _try_find(anchor, By.XPATH, xpath)
            if span:
                text = _normalize_text(getattr(span, "text", ""))
                if text and date_regex.search(text):
                    return date_regex.search(text).group(0)

        container = _nearest_container(anchor)

        # Prefer dates from end-of-section boxes when present
        filebox = _try_find(
            container,
            By.XPATH,
            ".//div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxend')][1]",
        )
        if filebox:
            spans = _try_find_all(filebox, By.XPATH, ".//span[normalize-space()]")
            for sp in spans:
                text = _normalize_text(getattr(sp, "text", ""))
                m = date_regex.search(text)
                if m:
                    return m.group(0)

        # Otherwise, scan all spans inside the container
        spans = _try_find_all(container, By.XPATH, ".//span[normalize-space()]")
        for sp in spans:
            text = _normalize_text(getattr(sp, "text", ""))
            m = date_regex.search(text)
            if m:
                return m.group(0)
        return None

    resources: List[Dict[str, Any]] = []
    for link in links or []:
        try:
            url = _safe_attr(link, "href")
            if not url:
                continue
            resource_type = classify_url(url)
            title = _extract_title(link, url)
            date_val = _extract_date(link)
            resources.append({
                "url": url,
                "title": title,
                "date": date_val,
                "type": resource_type,
            })
        except Exception as e:
            # Best-effort: skip problematic link but continue others
            logging.debug(f"Skipping link due to error: {e}")
            continue

    return resources