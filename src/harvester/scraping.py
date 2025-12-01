# src/harvester/scraping.py

import logging
import time
import os
import random
import re
from typing import Optional, List, Dict, Any, Tuple

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
from urllib.parse import quote, urljoin, urlparse

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-1.5-flash")

# Use a browser-like User-Agent to avoid 403s from some sites
BROWSER_HEADER = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# Domains that require authenticated scraping through Selenium to avoid 401/403 errors
SELENIUM_REQUIRED_DOMAINS: Tuple[str, ...] = (
    "docs.google.com",
    "drive.google.com",
    "gamma.app",
    "canva.com",
    "khanacademy.org",
)

# Domains that block unauthenticated HEAD probes; skip pre-flight content type checks.
HEAD_PROBE_BYPASS_DOMAINS: Tuple[str, ...] = (
    "gamma.app",
    "canva.com",
    "docs.google.com",
)

HEAD_FALLBACK_STATUS_CODES: Tuple[int, ...] = (
    401,
    403,
    405,
    429,
)

# Transcription scraping implementations moved to src/refinery/recording_processor.py


# --- Static Resource Helpers (Uses requests) ---

def _matches_domain(netloc: str, domain: str) -> bool:
    return netloc == domain or netloc.endswith(f".{domain}")


def _should_skip_head_probe(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False

    return any(_matches_domain(netloc, domain) for domain in HEAD_PROBE_BYPASS_DOMAINS)


def _infer_content_type_from_url(url: str) -> str:
    """
    Attempt to infer a reasonable content type based on the URL path/extension.
    Falls back to HTML to allow downstream handlers to attempt scraping.
    """
    sanitized = (url or "").split("?", 1)[0].split("#", 1)[0].lower()
    extension_map = {
        ".pdf": "application/pdf",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".ppt": "application/vnd.ms-powerpoint",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
    }
    for ext, mime in extension_map.items():
        if sanitized.endswith(ext):
            return f"{mime}; inferred-extension"
    return "text/html; head-probe-fallback"


def check_url_content_type(url: str) -> str:
    """Checks the Content-Type header of a URL using a HEAD request.

    For known sensitive domains that reject unauthenticated HEAD probes, we short-circuit to a
    scrapeable default to allow downstream Selenium-based handlers to proceed without failure.
    """
    logging.debug(f"Checking content type for URL: {url}")
    if not url or not url.startswith(("http://", "https://")):
        logging.warning(f"Invalid URL for content type check: {url}")
        return "unknown"
    if _should_skip_head_probe(url):
        logging.info(
            "   Skipping unauthenticated HEAD probe for sensitive domain; assuming HTML content."
        )
        # Return a string that downstream logic interprets as HTML-safe content.
        return "text/html; safe-probe-bypass"
    try:
        # Use HEAD request to check headers without downloading content
        resp = requests.head(url, timeout=15, allow_redirects=True, headers=BROWSER_HEADER)
        resp.raise_for_status()
        return (resp.headers.get("Content-Type", "")).lower()
    except RequestException as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code in HEAD_FALLBACK_STATUS_CODES:
            inferred_type = _infer_content_type_from_url(url)
            logging.warning(
                "   HEAD probe blocked for %s (HTTP %s). Falling back to inferred content type: %s",
                url,
                status_code,
                inferred_type,
            )
            return inferred_type
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

def _requires_authenticated_session(url: str) -> bool:
    """Determine if the URL belongs to a sensitive domain requiring an authenticated session."""
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False

    return any(_matches_domain(netloc, domain) for domain in SELENIUM_REQUIRED_DOMAINS)


# CRITICAL CONSTRAINT: For resources matching `SELENIUM_REQUIRED_DOMAINS`, the caller MUST ensure
# the authenticated `driver` object is passed, as the fallback to the `requests` library has been
# permanently disabled for these URLs.
def scrape_html_content(url: str, driver: Optional[webdriver.Chrome] = None) -> Optional[str]:
    """Scrapes and extracts the main readable text from a general webpage.

    Primary Method: GoogleDeepMind URL Extractor (Gemini URL Context).
    Fallback: Standard Selenium/Requests scraping.
    """
    logging.info(f"   Attempting to scrape HTML content from: {url}")

    # --- Primary Method: Gemini URL Context ---
    if genai and _GEMINI_API_KEY:
        try:
            logging.info("   Attempting Gemini URL extraction...")
            client = genai.Client(api_key=_GEMINI_API_KEY)

            # Use client.models.generate_content with the URL directly
            response = client.models.generate_content(
                model=_GEMINI_MODEL_NAME,
                contents=[
                    f"Extract the main article content from the following URL: {url}. "
                    "Return only the extracted text content, ignoring navigation, ads, and footers."
                ]
            )

            if response.text and len(response.text.strip()) > 50:
                 logging.info(f"   Successfully scraped content via Gemini ({len(response.text)} chars).")
                 return response.text
            else:
                 logging.warning("   Gemini URL extraction returned empty or too short content.")

        except Exception as e:
            logging.warning(f"   Gemini URL extraction failed: {e}. Falling back to standard scraping.")
    else:
        logging.info("   Gemini SDK not configured or available. Skipping primary method.")

    # --- Fallback: Standard Selenium/Requests scraping ---
    logging.debug(
        "   scrape_html_content fallback driver status: %s (id=%s)",
        "present" if driver is not None else "missing",
        getattr(driver, "session_id", "n/a") if driver is not None else "n/a",
    )
    requires_authenticated_session = _requires_authenticated_session(url)

    if requires_authenticated_session and driver is None:
        logging.critical(
            "   Authenticated Selenium driver required but not provided for sensitive URL: %s",
            url,
        )
        return None

    if requires_authenticated_session and driver is not None:
        try:
            logging.debug("   Using authenticated Selenium session for scraping.")
            driver.get(url)
            # Allow dynamic content to load briefly if needed
            time.sleep(random.uniform(1.0, 2.5))
            html_content = driver.page_source
        except WebDriverException as e:
            logging.error(
                "   Selenium scraping failed for sensitive URL %s: %s",
                url,
                e,
            )
            return None

        try:
            doc = ReadabilityDocument(html_content)
            main_content_html = doc.summary()
            soup = BeautifulSoup(main_content_html, "lxml")
            main_text = soup.get_text(separator="\n", strip=True)
            logging.info(f"   Successfully scraped HTML content via Selenium ({len(main_text)} chars).")
            return main_text
        except Exception as e:
            logging.warning(
                "   Failed to parse Selenium-captured HTML for %s: %s",
                url,
                e,
            )
            try:
                soup = BeautifulSoup(html_content, "lxml")
                fallback_text = soup.get_text(separator="\n", strip=True)
                if fallback_text:
                    logging.info(
                        "   Returning raw Selenium page text (%s chars) after readability failure.",
                        len(fallback_text),
                    )
                    return fallback_text
            except Exception as raw_exc:
                logging.warning(
                    "   Failed to extract raw text from Selenium page source for %s: %s",
                    url,
                    raw_exc,
                )
            return None

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

    # Granular Recording Classification
    if "zoom.us" in lower:
        return "RECORDING_ZOOM"

    # Check for specific Drive file/recording pattern (for recording categorization)
    if "drive.google.com/file" in lower:
        return "RECORDING_DRIVE"

    return "WEB_ARTICLE"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


_DATE_REGEX = re.compile(
    r"\b(?:\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s*\d{4})\b",
    re.IGNORECASE,
)


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


def _collect_anchor_metadata(container: WebElement) -> List[Tuple[str, bool]]:
    metadata: List[Tuple[str, bool]] = []
    anchors = _try_find_all(container, By.XPATH, ".//a[@href]")
    for anchor in anchors:
        try:
            href = (anchor.get_attribute("href") or "").strip()
        except StaleElementReferenceException:
            continue
        if not href:
            continue
        lower_href = href.lower()
        if lower_href.startswith("javascript:") or lower_href in {"#", ""}:
            continue
        try:
            icon_flag = _is_icon_link(anchor)
        except StaleElementReferenceException:
            icon_flag = False
        metadata.append((href, icon_flag))
    return metadata


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
        ".//ancestor-or-self::div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'filebox') and not(contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxrow'))][1]",
    ) or container

    anchor_metadata = _collect_anchor_metadata(filebox)
    for anchor_href, is_icon in anchor_metadata:
        if not is_icon:
            return _resolve_url(base_url, anchor_href)
    if anchor_metadata:
        return _resolve_url(base_url, anchor_metadata[0][0])

    # 3) Any sibling/descendant anchors
    fallback_metadata = _collect_anchor_metadata(container)
    for anchor_href, _is_icon in fallback_metadata:
        if anchor_href:
            return _resolve_url(base_url, anchor_href)
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
    # Prefer a nearby fileBoxend sibling following the fileBox container
    filebox = _try_find(
        container,
        By.XPATH,
        ".//ancestor-or-self::div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'filebox') and not(contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxrow'))][1]",
    ) or container
    fileboxend = _try_find(
        filebox,
        By.XPATH,
        "following-sibling::div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxend')][1]",
    ) or _try_find(
        container,
        By.XPATH,
        ".//div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxend')][1]",
    )
    if fileboxend:
        p_in_end = _try_find(fileboxend, By.XPATH, ".//p[normalize-space()][1] | .//a[normalize-space()][1]")
        if p_in_end and _normalize_text(getattr(p_in_end, "text", "")):
            return _normalize_text(getattr(p_in_end, "text", ""))
        # Fallback: use the text content of fileBoxend itself
        fb_text = _normalize_text(getattr(fileboxend, "text", ""))
        if fb_text:
            return fb_text

    # 4. aria-label or title attribute
    for attr in ("aria-label", "title"):
        val = _normalize_text(_safe_attr(anchor_or_container, attr))
        if val:
            return val

    # 5. Any nearby <p> or heading
    sib_p = _try_find(
        anchor_or_container,
        By.XPATH,
        "following-sibling::p[normalize-space()][1] | preceding-sibling::p[normalize-space()][1]",
    )
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
    filebox = _try_find(
        container,
        By.XPATH,
        ".//ancestor-or-self::div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'filebox')][1]",
    ) or container
    fileboxend = _try_find(
        filebox,
        By.XPATH,
        "following-sibling::div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxend')][1]",
    ) or _try_find(
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
        # Fallback: try the entire text content
        fb_text = _normalize_text(getattr(fileboxend, "text", ""))
        m = _DATE_REGEX.search(fb_text)
        if m:
            return m.group(0)

    # As a last resort, look for the first fileBoxend anywhere in the document
    try:
        doc_root = _try_find(container, By.XPATH, ".//ancestor::*[last()]") or container
        any_fbend = _try_find(
            doc_root,
            By.XPATH,
            ".//div[contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'fileboxend')][1]",
        )
        if any_fbend:
            text = _normalize_text(getattr(any_fbend, "text", ""))
            m = _DATE_REGEX.search(text)
            if m:
                return m.group(0)
    except Exception:
        pass

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
                logging.info(
                    f"Extracted resource -> url={url}, title={title}, date={date_val}, type={r_type}"
                )

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


# CRITICAL CONSTRAINT: For resources matching `SELENIUM_REQUIRED_DOMAINS`, the caller MUST ensure
# the authenticated `driver` object is passed, as the fallback to the `requests` library has been
# permanently disabled for these URLs.
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
            # Attempt to load the page (ensures driver session handles gated content)
            scraped = scrape_html_content(url, driver=driver)
            if scraped:
                content = scraped
            else:
                title = resource_metadata.get("title") or "Untitled Video"
                content = f"Video Title: {title}"

        elif resource_type == "WEB_ARTICLE":
            # Use HTTP-based scraper with authenticated fallback for sensitive domains
            scraped = scrape_html_content(url, driver=driver)
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
                        page_texts = [
                            p.get("text", "") for p in (extracted_pages or []) if isinstance(p, dict)
                        ]
                        content = "\n\n".join([t for t in page_texts if t]) or "(No text extracted from PDF)"
                except Exception as e:
                    content = f"PDF processing failed: {e}"
                finally:
                    try:
                        if tmp_path and os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass

        elif resource_type in {
            "OFFICE_DOCUMENT",
            "GOOGLE_DOCS",
            "GOOGLE_SHEETS",
            "GOOGLE_SLIDES",
            "OFFICE_ONLINE_VIEWER",
        }:
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

        elif resource_type in {"ZOOM_RECORDING", "RECORDING_ZOOM"}:
            # EXECUTE the remote Zoom transcription logic
            try:
                from src.refinery.recording_processor import extract_transcript  # type: ignore
                content = extract_transcript(driver, url, resource_type)
                if not content:
                    content = "(Zoom transcript extraction failed or returned no text)"
            except Exception as e:
                logging.error(f"Zoom transcription failed: {e}")
                content = f"Zoom transcription failure: {e}"

        elif resource_type in {"DRIVE_RECORDING", "RECORDING_DRIVE"}:
            try:
                content = extract_transcript(driver, url, resource_type)
                if not content:
                    content = "(Drive transcript extraction failed or returned no text)"
            except Exception as e:
                logging.error(f"Drive transcription failed: {e}")
                content = f"Drive transcription failure: {e}"

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