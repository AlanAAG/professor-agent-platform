import html
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
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
from urllib.parse import urlparse, urlsplit, urljoin

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# Import harvester-side config and navigation helpers via absolute package path
from src.harvester import config
from src.harvester.navigation import safe_find, safe_find_all, safe_click

_TIMEDTEXT_URL_REGEX = re.compile(
    r"https://drive\.google\.com[^\"'`<>\s]*timedtext[^\"'`<>\s]*",
    re.IGNORECASE,
)

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")

_ENABLE_FALLBACK = os.environ.get("ENABLE_WHISPER_FALLBACK", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_FALLBACK_MAX_DOWNLOAD_MB = float(os.environ.get("WHISPER_MAX_DOWNLOAD_MB", "400"))
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

def _decode_drive_timedtext_url(raw_url: Optional[str]) -> Optional[str]:
    if not raw_url:
        return None
    decoded = raw_url
    for needle, replacement in (
        ("\\u003d", "="),
        ("\\u0026", "&"),
        ("\\u002F", "/"),
        ("\\u003f", "?"),
        ("\\x3d", "="),
        ("\\x26", "&"),
        ("\\/", "/"),
    ):
        decoded = decoded.replace(needle, replacement)
    decoded = html.unescape(decoded)
    return decoded


def _locate_drive_caption_url(driver: webdriver.Chrome, wait_timeout: int) -> Optional[str]:
    """Return the first timedtext URL embedded in the Drive preview page."""
    try:
        WebDriverWait(driver, wait_timeout).until(
            lambda d: bool(getattr(d, "current_url", "") or d.find_elements(By.TAG_NAME, "video"))
        )
    except Exception:
        pass

    script = """
        return (function () {
          const pattern = /https:\\/\\/drive\\.google\\.com[^"'`\\s]*timedtext[^"'`\\s]*/i;

          const inspect = (text) => {
            if (!text) return null;
            const match = text.match(pattern);
            if (match && match[0]) {
              return match[0];
            }
            return null;
          };

          const probeTracks = (obj, depth = 0) => {
            if (!obj || typeof obj !== "object" || depth > 3) return null;
            const directKeys = ["captionsUrl", "timedtextUrl", "url", "baseUrl"];
            for (const key of directKeys) {
              if (typeof obj[key] === "string") {
                const found = inspect(obj[key]);
                if (found) return found;
              }
            }
            const collections = obj.captionTracks || obj.captionsTracks || obj.captions || obj.tracks;
            if (Array.isArray(collections)) {
              for (const entry of collections) {
                const candidate = probeTracks(entry, depth + 1);
                if (candidate) return candidate;
              }
            }
            return null;
          };

          const globalCandidates = [
            window._DRIVE_ivd,
            window._DRIVE_video_data,
            window._DRIVE_translations,
            window.WIZ_global_data,
            window.APP_INITIALIZATION_STATE,
            window.OZ_initData,
          ];
          for (const candidate of globalCandidates) {
            const found = probeTracks(candidate);
            if (found) return found;
          }

          const scripts = Array.from(document.scripts || []);
          for (const node of scripts) {
            const text = node.text || node.innerHTML || "";
            const found = inspect(text);
            if (found) return found;
          }

          const serialized = document.documentElement ? document.documentElement.innerHTML : "";
          return inspect(serialized) || null;
        })();
    """
    try:
        captured: Optional[str] = driver.execute_script(script)
    except WebDriverException as exc:
        logging.debug("      Failed to execute caption URL probe: %s", exc)
        captured = None

    if not captured and getattr(driver, "page_source", None):
        match = _TIMEDTEXT_URL_REGEX.search(driver.page_source or "")
        if match:
            captured = match.group(0)

    return _decode_drive_timedtext_url(captured)


def _fetch_drive_timedtext_payload(
    driver: webdriver.Chrome,
    captions_url: str,
) -> Optional[Dict[str, Any]]:
    script = """
        var url = arguments[0];
        var callback = arguments[1];
        if (!url) {
          callback({ ok: false, status: 0, error: "missing url" });
          return;
        }
        fetch(url, { credentials: "include" })
          .then((resp) => resp.text().then((text) => ({
              ok: resp.ok,
              status: resp.status,
              text
          })))
          .then((result) => {
            callback(result);
          })
          .catch((err) => {
            callback({ ok: false, status: 0, error: String(err && err.message ? err.message : err) });
          });
    """
    try:
        response = driver.execute_async_script(script, captions_url)
    except WebDriverException as exc:
        logging.error("      Timedtext fetch via browser failed: %s", exc)
        return None

    if not isinstance(response, dict):
        logging.warning("      Timedtext fetch returned unexpected payload type: %s", type(response))
        return None

    if not response.get("ok"):
        logging.warning(
            "      Timedtext request failed (status=%s, error=%s)",
            response.get("status"),
            response.get("error"),
        )
        return None

    text = response.get("text") or ""
    if not text.strip():
        logging.warning("      Timedtext response empty.")
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logging.error("      Failed to decode timedtext JSON: %s", exc)
        return None


def _parse_drive_timedtext(payload: Dict[str, Any]) -> str:
    events: List[Dict[str, Any]] = payload.get("events") or []
    lines: List[str] = []
    for event in events:
        segs = event.get("segs") or []
        if not isinstance(segs, list):
            continue
        words: List[str] = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            text = (seg.get("utf8") or "").strip()
            if not text:
                continue
            words.append(text)
        if words:
            assembled = " ".join(words).strip()
            if assembled:
                lines.append(assembled)
    transcript = "\n".join(lines).strip()
    return transcript


def scrape_drive_transcript_content(driver: webdriver.Chrome) -> str:
    """Scrape transcript content from a Google Drive recording preview page via timedtext API."""
    logging.info("   Attempting Google Drive transcript scrape (timedtext)...")
    wait_timeout = getattr(config.SETTINGS, "wait_timeout", 30)
    captions_url = _locate_drive_caption_url(driver, wait_timeout)
    if not captions_url:
        logging.warning("      Unable to locate timedtext URL on Drive page.")
        return ""

    payload = _fetch_drive_timedtext_payload(driver, captions_url)
    if not payload:
        return ""

    transcript = _parse_drive_timedtext(payload)
    if transcript:
        logging.info("   Successfully scraped Google Drive transcript (%s chars).", len(transcript))
        return transcript

    logging.warning("      Timedtext payload parsed but produced no transcript.")
    return ""


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


def _can_attempt_gemini_fallback() -> bool:
    if not _ENABLE_FALLBACK:
        logging.info("   Fallback disabled via ENABLE_WHISPER_FALLBACK.")
        return False
    if genai is None:
        logging.warning("   Google Generative AI SDK not installed; cannot run Gemini fallback.")
        return False
    if not _GEMINI_API_KEY:
        logging.warning("   GEMINI_API_KEY is not set; skipping Gemini fallback.")
        return False
    return True


def _build_session_from_driver(driver: webdriver.Chrome) -> requests.Session:
    session = requests.Session()
    referer = getattr(driver, "current_url", "") or ""
    headers = {
        "User-Agent": _DEFAULT_USER_AGENT,
        "Referer": referer,
        "Accept": "*/*",
    }
    try:
        ua = driver.execute_script("return navigator.userAgent || ''")  # type: ignore[attr-defined]
        if isinstance(ua, str) and ua.strip():
            headers["User-Agent"] = ua.strip()
    except WebDriverException:
        pass
    session.headers.update(headers)
    try:
        for cookie in driver.get_cookies():
            name = cookie.get("name")
            value = cookie.get("value")
            if not name or value is None:
                continue
            domain = cookie.get("domain")
            path = cookie.get("path", "/")
            session.cookies.set(name, value, domain=domain, path=path)
    except WebDriverException:
        pass
    return session


def _heuristic_zoom_download_url(url: str) -> Optional[str]:
    if not url:
        return None
    parts = urlsplit(url)
    path = parts.path
    if "/rec/share/" in path:
        path = path.replace("/rec/share/", "/rec/download/", 1)
    elif "/rec/play/" in path:
        path = path.replace("/rec/play/", "/rec/download/", 1)
    query = parts.query
    if "download=1" not in query.split("&"):
        query = f"{query}&download=1" if query else "download=1"
    rebuilt = parts._replace(path=path, query=query)
    return rebuilt.geturl()


def _resolve_zoom_download_url(driver: webdriver.Chrome, url: str) -> Optional[str]:
    selectors = [
        "a[href*='/rec/download']",
        "a[data-event='download']",
        "a[download]",
        "button[data-callback='download']",
        "button[title*='Download']",
    ]
    script = """
        const selectors = arguments[0];
        for (const selector of selectors) {
            const node = document.querySelector(selector);
            if (!node) continue;
            if (node.href) return node.href;
            const href = node.getAttribute('href');
            if (href) return href;
        }
        if (window?.cloudRecordingPlayer?.downloadInfo?.downloadUrl) {
            return window.cloudRecordingPlayer.downloadInfo.downloadUrl;
        }
        if (window?.cloudSettings?.downloadUrl) {
            return window.cloudSettings.downloadUrl;
        }
        if (window?.preload?.downloadUrl) {
            return window.preload.downloadUrl;
        }
        return null;
    """
    try:
        candidate = driver.execute_script(script, selectors)
    except WebDriverException:
        candidate = None
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return _heuristic_zoom_download_url(getattr(driver, "current_url", "") or url)


def _extract_drive_file_id(url: str) -> Optional[str]:
    if not url:
        return None
    match = re.search(r"/file/d/([^/]+)/", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([^&]+)", url)
    if match:
        return match.group(1)
    return None


def _resolve_drive_download_url(driver: webdriver.Chrome, url: str) -> Optional[str]:
    selectors = [
        "a[aria-label='Download']",
        "a[role='menuitem'][href*='export=download']",
        "a[href*='uc?export=download']",
        "a[download]",
    ]
    script = """
        const selectors = arguments[0];
        for (const selector of selectors) {
            const node = document.querySelector(selector);
            if (!node) continue;
            if (node.href) return node.href;
            const href = node.getAttribute('href');
            if (href) return href;
        }
        return null;
    """
    candidate = None
    try:
        candidate = driver.execute_script(script, selectors)
    except WebDriverException:
        candidate = None
    if isinstance(candidate, str) and candidate.strip():
        href = candidate.strip()
        current = getattr(driver, "current_url", "") or url
        if href.startswith(("http://", "https://")):
            return href
        try:
            return urljoin(current, href)
        except Exception:
            return href
    file_id = _extract_drive_file_id(getattr(driver, "current_url", "") or url)
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return None


def _download_recording_media(driver: webdriver.Chrome, download_url: str) -> Optional[str]:
    if not download_url:
        return None
    session = _build_session_from_driver(driver)
    target_dir = os.path.join(config.SETTINGS.downloads_dir, "recordings")
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError as exc:
        logging.error("   Unable to create recordings download directory: %s", exc)
        return None
    parsed = urlsplit(download_url)
    _, ext = os.path.splitext(parsed.path)
    if not ext or len(ext) > 5:
        ext = ".mp4"
    filename = f"recording_{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(target_dir, filename)
    max_bytes = int(_FALLBACK_MAX_DOWNLOAD_MB * 1024 * 1024)
    try:
        resp = session.get(download_url, stream=True, timeout=(15, 300))
        resp.raise_for_status()

        # Check for Google Drive interstitial warning (HTML instead of video)
        content_type = resp.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            logging.warning("   Download returned HTML content. Checking for confirmation token...")
            content = resp.text  # Consumes the stream
            resp.close()

            soup = BeautifulSoup(content, "html.parser")
            confirm_href = None
            for a in soup.find_all("a", href=True):
                if "confirm=" in a["href"]:
                    confirm_href = a["href"]
                    break

            if confirm_href:
                confirm_href = urljoin(download_url, confirm_href)
                logging.info("   Found confirmation token. Retrying download...")
                resp = session.get(confirm_href, stream=True, timeout=(15, 300))
                resp.raise_for_status()
            else:
                logging.error("   Download returned HTML but no confirmation token found.")
                return None

        # Check again in case the retry also returned HTML
        if "text/html" in resp.headers.get("Content-Type", "").lower():
            logging.error("   Download (or retry) returned HTML content; aborting.")
            resp.close()
            return None

        with resp:
            written = 0
            with open(file_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
                    if written > max_bytes:
                        logging.error(
                            "   Recording download exceeded %s MB limit; aborting fallback.",
                            _FALLBACK_MAX_DOWNLOAD_MB,
                        )
                        fh.close()
                        os.remove(file_path)
                        return None

        # Verify file validity (size check)
        if os.path.getsize(file_path) < 10 * 1024:  # 10KB
            logging.error("   Downloaded file is too small (<10KB); treating as failure.")
            os.remove(file_path)
            return None

    except RequestException as exc:
        logging.error("   Failed to download recording for fallback: %s", exc)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass
        return None
    except OSError as exc:
        logging.error("   Failed to persist recording for fallback: %s", exc)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass
        return None

    return file_path


def _transcribe_with_gemini(file_path: str) -> str:
    if not _can_attempt_gemini_fallback():
        return ""

    logging.info("   Gemini fallback: uploading audio/video to Gemini...")
    client = genai.Client(api_key=_GEMINI_API_KEY)
    video_file = None
    try:
        # Upload the file
        video_file = client.files.upload(file=file_path)

        # Wait for processing
        logging.info("   Gemini fallback: waiting for file processing...")
        # Add simple timeout mechanism to avoid infinite loops
        max_retries = 60  # Wait up to 2 minutes (60 * 2s)
        retries = 0

        # Check state. Handle both string and enum cases for safety across SDK versions
        def get_state_str(f):
            if hasattr(f, "state"):
                s = f.state
                if hasattr(s, "name"):
                    return s.name
                return str(s)
            return ""

        while get_state_str(video_file) == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)
            retries += 1
            if retries >= max_retries:
                logging.error("   Gemini file processing timed out.")
                return ""

        if get_state_str(video_file) == "FAILED":
            logging.error("   Gemini file processing failed.")
            return ""

        logging.info("   Gemini fallback: requesting transcription from %s...", _GEMINI_MODEL_NAME)

        # Prompt for transcription
        prompt = "Transcribe the audio in this file into text. Provide only the transcription, no introductory text."

        response = client.models.generate_content(
            model=_GEMINI_MODEL_NAME,
            contents=[video_file, prompt]
        )

        return response.text if response.text else ""

    except Exception as exc:
        logging.error("   Gemini transcription failed: %s", exc)
        return ""
    finally:
        # Cleanup file on Gemini side
        if video_file:
            try:
                logging.info("   Gemini fallback: deleting remote file %s", video_file.name)
                client.files.delete(name=video_file.name)
            except Exception as e:
                logging.warning("   Failed to delete remote file on Gemini: %s", e)


def _attempt_fallback(driver: webdriver.Chrome, url: str, resource_type: str) -> str:
    if not _can_attempt_gemini_fallback():
        return ""
    normalized = (resource_type or "").upper()
    download_url: Optional[str] = None
    if "ZOOM" in normalized:
        download_url = _resolve_zoom_download_url(driver, url)
    elif "DRIVE" in normalized:
        download_url = _resolve_drive_download_url(driver, url)
    else:
        logging.warning("   Fallback does not support resource type: %s", resource_type)
        return ""
    if not download_url:
        logging.warning("   Fallback: unable to determine download URL for %s", url)
        return ""
    media_path = _download_recording_media(driver, download_url)
    if not media_path:
        return ""
    try:
        transcript = _transcribe_with_gemini(media_path)
        if transcript:
            logging.info("   Fallback succeeded (%s chars).", len(transcript))
        else:
            logging.warning("   Fallback produced no text.")
        return transcript
    finally:
        try:
            if os.path.exists(media_path):
                os.remove(media_path)
        except OSError:
            pass


def extract_transcript(driver: webdriver.Chrome, url: str, resource_type: str) -> str:
    """Public entry point to selectively extract transcript content from a recording URL.

    - For ZOOM_RECORDING: open in a new tab and scrape transcript content.
    - For DRIVE_RECORDING: open in a new tab and retrieve transcript via timedtext endpoint.
    """
    original = driver.current_window_handle
    new_handle = None
    transcript_text = ""
    normalized_type = (resource_type or "").upper()

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
        if normalized_type in {"ZOOM_RECORDING", "RECORDING_ZOOM"}:
            transcript_text = scrape_zoom_transcript_content(driver)
        elif normalized_type in {"DRIVE_RECORDING", "RECORDING_DRIVE"}:
            transcript_text = scrape_drive_transcript_content(driver)
        else:
            logging.warning(f"   Unknown recording type: {resource_type}. Skipping primary scrape.")

        if not transcript_text:
            logging.info("   Primary transcript scrape empty; attempting fallback.")
            transcript_text = _attempt_fallback(driver, url, normalized_type) or ""

        return transcript_text

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
