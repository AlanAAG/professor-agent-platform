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
from urllib.parse import quote

# Use a browser-like User-Agent to avoid 403s from some sites
BROWSER_HEADER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# Transcription scraping implementations moved to src/refinery/recording_processor.py


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
    # Granular Recording Classification
    if "zoom.us" in lower:
        return "ZOOM_RECORDING"
    if "drive.google.com/file" in lower and ("view?usp=drive_link" in lower or "preview" in lower):
        return "DRIVE_RECORDING"

    # Broad fallback for other recording links
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

        elif resource_type == "OFFICE_DOCUMENT":
            # Non-download strategy: use web viewers for Office/Google Docs/Sheets/Slides
            try:
                from src.refinery.document_processing import extract_drive_content  # type: ignore
            except Exception as e:  # pragma: no cover - import safety
                logging.error(f"Failed to import in-browser document extractor: {e}")
                content = f"Office document content extraction failed (import): {e}"
            else:
                viewer_url = url
                lower = (url or "").lower()
                # If direct office file link, route via Office Online viewer to avoid download
                if (
                    (lower.endswith((".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx")))
                    and ("docs.google.com" not in lower and "drive.google.com" not in lower)
                ):
                    # Office online viewer
                    viewer_url = f"https://view.officeapps.live.com/op/view.aspx?src={quote(url, safe='')}"

                content = extract_drive_content(driver, viewer_url, resource_type)
                if not content:
                    content = "(No readable content found in document viewer)"

        elif resource_type == "ZOOM_RECORDING":
            # EXECUTE the remote Zoom transcription logic
            try:
                from src.refinery.recording_processor import extract_transcript  # type: ignore
                content = extract_transcript(driver, url, resource_type)
                if not content:
                    content = "(Zoom transcript extraction failed or returned no text)"
            except Exception as e:
                logging.error(f"Zoom transcription failed: {e}")
                content = f"Zoom transcription failure: {e}"

        elif resource_type == "DRIVE_RECORDING":
            # Hard skip the expensive Drive operation as requested
            logging.warning(
                f"DRIVE_RECORDING: Skipping complex scraping logic as requested."
            )
            content = "Transcription skipped (Drive Recording)."

        elif resource_type == "RECORDING_LINK":
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