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
from urllib.parse import quote, urljoin

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

    Enhanced types:
      - GOOGLE_DOCS, GOOGLE_SHEETS, GOOGLE_SLIDES
      - OFFICE_ONLINE_VIEWER (view.officeapps.live.com)
      - OFFICE_DOCUMENT (direct .docx/.pptx/.xlsx links)
      - RECORDING_DRIVE (drive.google.com), RECORDING_ZOOM (zoom.us)
      - YOUTUBE_VIDEO, PDF_DOCUMENT, WEB_ARTICLE
    """
    if not url:
        return "WEB_ARTICLE"
    lower = url.lower()

    if "youtube.com/watch" in lower or "youtu.be/" in lower:
        return "YOUTUBE_VIDEO"
    if lower.endswith(".pdf"):
        return "PDF_DOCUMENT"

    if "docs.google.com/document" in lower:
        return "GOOGLE_DOCS"
    if "docs.google.com/spreadsheets" in lower:
        return "GOOGLE_SHEETS"
    if "docs.google.com/presentation" in lower:
        return "GOOGLE_SLIDES"

    if "view.officeapps.live.com" in lower:
        return "OFFICE_ONLINE_VIEWER"
    if lower.endswith((".pptx", ".ppt", ".doc", ".docx", ".xls", ".xlsx")):
        return "OFFICE_DOCUMENT"

    if "drive.google.com/file" in lower:
        return "RECORDING_DRIVE"
    if "zoom.us" in lower:
        return "RECORDING_ZOOM"

    return "WEB_ARTICLE"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


_DATE_REGEX = re.compile(r"\b(?:\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4})\b",
                         re.IGNORECASE)


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


def _nearest_container(anchor_or_container: WebElement) -> WebElement:
    container = _try_find(
        anchor_or_container,
        By.XPATH,
        "./ancestor::*[self::li or self::div or self::section or self::article][1]",
    )
    return container or anchor_or_container


def _is_icon_link(anchor: WebElement) -> bool:
    try:
        # Heuristic: no meaningful text and contains only svg/img
        text = _normalize_text(getattr(anchor, "text", ""))
        if text:
            return False
        svg_or_img = _try_find(anchor, By.XPATH, ".//*[self::svg or self::img]")
        return bool(svg_or_img)
    except Exception:
        return False


def _resolve_url(base_url: str, href: str) -> str:
    if not href:
        return ""
    if href.startswith(("http://", "https://")):
        return href
    return urljoin(base_url, href)


def _extract_url_robust(anchor_or_container: WebElement, base_url: str) -> str:
    # 1) Direct href on the given element
    href = _safe_attr(anchor_or_container, "href")
    if href:
        return _resolve_url(base_url, href)

    # 2) Search in nearest .fileBox container
    container = _nearest_container(anchor_or_container)
    filebox = _try_find(
        container,
        By.XPATH,
        ".//ancestor-or-self::div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'filebox')][1]",
    ) or container

    anchors = _try_find_all(filebox, By.XPATH, ".//a[@href]")
    # Prefer first visible and non-icon anchor; if none, fall back to first anchor
    for a in anchors:
        href = _safe_attr(a, "href")
        if not href:
            continue
        if _is_icon_link(a):
            continue
        return _resolve_url(base_url, href)
    if anchors:
        return _resolve_url(base_url, _safe_attr(anchors[0], "href"))

    # 3) Any sibling/descendant anchors
    anchors = _try_find_all(container, By.XPATH, ".//a[@href]")
    for a in anchors:
        href = _safe_attr(a, "href")
        if href:
            return _resolve_url(base_url, href)
    return ""


def _extract_title_robust(anchor_or_container: WebElement, url: str) -> str:
    # 1. Anchor text
    anchor_text = _normalize_text(getattr(anchor_or_container, "text", ""))
    if anchor_text and anchor_text.lower() != (url or "").lower():
        return anchor_text

    # 2. <p> inside fileContentCol
    container = _nearest_container(anchor_or_container)
    p_in_content = _try_find(
        container,
        By.XPATH,
        ".//div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'filecontentcol')]//p[normalize-space()][1]",
    )
    if p_in_content and _normalize_text(getattr(p_in_content, "text", "")):
        return _normalize_text(getattr(p_in_content, "text", ""))

    # 3. <p> in fileBoxend (longer descriptive title)
    fileboxend = _try_find(
        container,
        By.XPATH,
        ".//div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxend')][1]",
    )
    if fileboxend:
        p_in_end = _try_find(fileboxend, By.XPATH, ".//p[normalize-space()][1] | .//a[normalize-space()][1]")
        if p_in_end and _normalize_text(getattr(p_in_end, "text", "")):
            return _normalize_text(getattr(p_in_end, "text", ""))

    # 4. aria-label or title attribute
    for attr in ("aria-label", "title"):
        val = _normalize_text(_safe_attr(anchor_or_container, attr))
        if val:
            return val

    # 5. Any nearby <p> or heading
    sib_p = _try_find(anchor_or_container, By.XPATH, "following-sibling::p[normalize-space()][1] | preceding-sibling::p[normalize-space()][1]")
    if sib_p and _normalize_text(getattr(sib_p, "text", "")):
        return _normalize_text(getattr(sib_p, "text", ""))

    headings = _try_find_all(container, By.XPATH, ".//h1|.//h2|.//h3|.//h4|.//h5|.//h6")
    for h in headings:
        h_text = _normalize_text(getattr(h, "text", ""))
        if h_text:
            return h_text

    # 6. Parent container's visible text
    container_text = _normalize_text(getattr(container, "text", ""))
    if container_text:
        return container_text

    # 7. Fallback
    return url or "Untitled Resource"


def _extract_date_robust(anchor_or_container: WebElement) -> Optional[str]:
    # 1. <span> in fileBoxend
    container = _nearest_container(anchor_or_container)
    fileboxend = _try_find(
        container,
        By.XPATH,
        ".//div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxend')][1]",
    )
    if fileboxend:
        spans = _try_find_all(fileboxend, By.XPATH, ".//span[normalize-space()]")
        for sp in spans:
            text = _normalize_text(getattr(sp, "text", ""))
            m = _DATE_REGEX.search(text)
            if m:
                return m.group(0)

    # 2. <span> in fileContentCol
    spans = _try_find_all(
        container,
        By.XPATH,
        ".//div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'filecontentcol')]//span[normalize-space()]",
    )
    for sp in spans:
        text = _normalize_text(getattr(sp, "text", ""))
        m = _DATE_REGEX.search(text)
        if m:
            return m.group(0)

    # 3. Any text matching regex near element
    spans = _try_find_all(container, By.XPATH, ".//span|.//p|.//*")
    for sp in spans:
        text = _normalize_text(getattr(sp, "text", ""))
        m = _DATE_REGEX.search(text)
        if m:
            return m.group(0)
    return None


def process_resource_links(
    driver: webdriver.Chrome,
    links: List[WebElement],
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Process a list of WebElements into structured resource metadata.

    Returns items with keys: url, title, date, type
    """
    resources: List[Dict[str, Any]] = []
    base_url = getattr(config, "BASE_URL", "")

    for item in links or []:
        try:
            # URL extraction with multiple strategies
            url = _extract_url_robust(item, base_url)
            if not url:
                if debug:
                    logging.debug("No URL found for element; skipping")
                continue

            # Title and date using robust helpers
            title = _extract_title_robust(item, url)
            if not title:
                logging.warning(f"Missing title for URL: {url}")
                title = "Untitled Resource"

            date_val = _extract_date_robust(item)
            if not date_val:
                logging.warning(f"No date found near URL: {url}")

            r_type = classify_url(url)

            if debug:
                logging.info(f"Extracted resource -> url={url}, title={title}, date={date_val}, type={r_type}")

            resources.append({
                "url": url,
                "title": title,
                "date": date_val,
                "type": r_type,
            })
        except StaleElementReferenceException:
            logging.warning("Encountered stale element reference; skipping one link")
            continue
        except Exception as e:
            logging.warning(f"Failed to process a resource link: {e}")
            continue

    return resources


def scrape_and_refine_resource(driver: webdriver.Chrome, resource_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Routes resource scraping based on type and returns refined data.

    Prioritizes in-browser scraping for Google Drive/Docs/Sheets/Slides and direct Office links
    to avoid downloads. PDF is the sole exception where a download may be used for reliability.
    """
    url = resource_metadata.get("url") or ""
    resource_type = resource_metadata.get("type") or classify_url(url)
    content: str | None = None
    logging.info(f"Processing resource type: {resource_type} for URL: {url[:80]}...")

    try:
        if resource_type == "YOUTUBE_VIDEO":
            # Lightweight: rely on previously extracted title
            title = resource_metadata.get("title") or "Untitled Video"
            content = f"Video Title: {title}"

        elif resource_type == "WEB_ARTICLE":
            # Use lightweight HTTP-based scraper
            scraped = scrape_html_content(url)
            content = scraped or (resource_metadata.get("title") or "")

        elif resource_type == "PDF_DOCUMENT":
            # Only allowed download exception
            try:
                from src.refinery.pdf_processing import process_pdf  # type: ignore
            except Exception as e:  # pragma: no cover - import safety
                logging.error(f"Failed to import PDF processor: {e}")
                content = f"PDF processing unavailable: {e}"
            else:
                # Best-effort: download then process, cleanup afterward
                tmp_path: Optional[str] = None
                try:
                    filename = f"temp_{int(time.time())}.pdf"
                    tmp_dir = os.path.join("/tmp", "refinery_pdfs")
                    tmp_path = download_file(url, tmp_dir, filename)
                    if not tmp_path:
                        content = "PDF download failed."
                    else:
                        extracted_pages = process_pdf(tmp_path)  # existing API expects a file path
                        # Flatten to simple text for now
                        page_texts = [p.get("text", "") for p in (extracted_pages or []) if isinstance(p, dict)]
                        content = "\n\n".join([t for t in page_texts if t]) or "(No text extracted from PDF)"
                except Exception as e:
                    content = f"PDF processing failed: {e}"
                finally:
                    try:
                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass

        elif resource_type in {"OFFICE_DOCUMENT", "GOOGLE_DOCS", "GOOGLE_SHEETS", "GOOGLE_SLIDES", "OFFICE_ONLINE_VIEWER"}:
            # Use web viewers (no downloads)
            try:
                from src.refinery.document_processing import extract_drive_content  # type: ignore
            except Exception as e:  # pragma: no cover
                logging.error(f"Failed to import in-browser document extractor: {e}")
                content = f"Office document content extraction failed (import): {e}"
            else:
                # Determine specific doc type
                lower = (url or "").lower()
                if "docs.google.com/document" in lower:
                    doc_type = "google_docs"
                elif "docs.google.com/spreadsheets" in lower:
                    doc_type = "google_sheets"
                elif "docs.google.com/presentation" in lower:
                    doc_type = "google_slides"
                else:
                    doc_type = "office_document"

                content = extract_drive_content(driver, url, doc_type)
                if not content:
                    content = "(No readable content found in document viewer)"

        elif resource_type in {"RECORDING_LINK", "RECORDING_DRIVE", "RECORDING_ZOOM"}:
            # Intentionally skip transcription to reduce costs
            logging.warning(
                f"Skipping transcription for recording link: {url} (Too resource-intensive)."
            )
            content = "Transcription skipped (Recording Link)."

        else:
            content = "Unknown or unhandled resource type."

    except Exception as e:
        logging.error(f"Error during resource processing: {e}")
        if not content:
            content = f"Processing failure: {e}"

    resource_metadata["extracted_content"] = content or ""
    return resource_metadata