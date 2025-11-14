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
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

# Import harvester-side config and navigation helpers via absolute package path
from src.harvester import config
from src.harvester.navigation import safe_find, safe_find_all, safe_click

_TIMEDTEXT_URL_REGEX = re.compile(
    r"https://drive\.google\.com[^\"'`<>\s]*timedtext[^\"'`<>\s]*",
    re.IGNORECASE,
)

_WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "whisper-1")
_ENABLE_WHISPER_FALLBACK = os.environ.get("ENABLE_WHISPER_FALLBACK", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_WHISPER_MAX_DOWNLOAD_MB = float(os.environ.get("WHISPER_MAX_DOWNLOAD_MB", "400"))
_OPENAI_AUDIO_CLIENT: Optional["OpenAI"] = None
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


def _can_attempt_whisper_fallback() -> bool:
    if not _ENABLE_WHISPER_FALLBACK:
        logging.info("   Whisper fallback disabled via ENABLE_WHISPER_FALLBACK.")
        return False
    if OpenAI is None:
        logging.warning("   OpenAI SDK not installed; cannot run Whisper fallback.")
        return False
    if not os.environ.get("OPENAI_API_KEY"):
        logging.warning("   OPENAI_API_KEY is not set; skipping Whisper fallback.")
        return False
    return True


def _get_openai_audio_client() -> Optional["OpenAI"]:
    global _OPENAI_AUDIO_CLIENT
    if _OPENAI_AUDIO_CLIENT is not None:
        return _OPENAI_AUDIO_CLIENT
    if OpenAI is None:
        return None
    try:
        _OPENAI_AUDIO_CLIENT = OpenAI()
    except Exception as exc:  # pragma: no cover - initialization failure is rare
        logging.error("   Failed to initialize OpenAI client for Whisper fallback: %s", exc)
        _OPENAI_AUDIO_CLIENT = None
    return _OPENAI_AUDIO_CLIENT


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
    max_bytes = int(_WHISPER_MAX_DOWNLOAD_MB * 1024 * 1024)
    try:
        with session.get(download_url, stream=True, timeout=(15, 300)) as resp:
            resp.raise_for_status()
            written = 0
            with open(file_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    written += len(chunk)
                    if written > max_bytes:
                        logging.error(
                            "   Recording download exceeded %s MB limit; aborting Whisper fallback.",
                            _WHISPER_MAX_DOWNLOAD_MB,
                        )
                        fh.close()
                        os.remove(file_path)
                        return None
    except RequestException as exc:
        logging.error("   Failed to download recording for Whisper fallback: %s", exc)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass
        return None
    except OSError as exc:
        logging.error("   Failed to persist recording for Whisper fallback: %s", exc)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass
        return None
    return file_path


def _transcribe_with_whisper(file_path: str) -> str:
    client = _get_openai_audio_client()
    if client is None:
        return ""
    logging.info("   Whisper fallback: sending audio to %s ...", _WHISPER_MODEL_NAME)
    try:
        with open(file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=_WHISPER_MODEL_NAME,
                file=audio_file,
            )
    except Exception as exc:  # pragma: no cover - network failure
        logging.error("   Whisper transcription failed: %s", exc)
        return ""
    text = ""
    if isinstance(response, dict):
        text = response.get("text") or ""
    else:
        text = getattr(response, "text", "") or ""
    return text.strip()


def _attempt_whisper_fallback(driver: webdriver.Chrome, url: str, resource_type: str) -> str:
    if not _can_attempt_whisper_fallback():
        return ""
    normalized = (resource_type or "").upper()
    download_url: Optional[str] = None
    if "ZOOM" in normalized:
        download_url = _resolve_zoom_download_url(driver, url)
    elif "DRIVE" in normalized:
        download_url = _resolve_drive_download_url(driver, url)
    else:
        logging.warning("   Whisper fallback does not support resource type: %s", resource_type)
        return ""
    if not download_url:
        logging.warning("   Whisper fallback: unable to determine download URL for %s", url)
        return ""
    media_path = _download_recording_media(driver, download_url)
    if not media_path:
        return ""
    try:
        transcript = _transcribe_with_whisper(media_path)
        if transcript:
            logging.info("   Whisper fallback succeeded (%s chars).", len(transcript))
        else:
            logging.warning("   Whisper fallback produced no text.")
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
            logging.info("   Primary transcript scrape empty; attempting Whisper fallback.")
            transcript_text = _attempt_whisper_fallback(driver, url, normalized_type) or ""

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
