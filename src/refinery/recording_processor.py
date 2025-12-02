import html
import json
import logging
import os
import re
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional

from imageio_ffmpeg import get_ffmpeg_exe
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
from moviepy import VideoFileClip

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
    match = re.search(r"/file/d/([^/&?]+)", url)
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


def _download_recording_media(driver: webdriver.Chrome, download_url: str, original_url: Optional[str] = None) -> Optional[str]:
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

    # Track if API/Request method failed with HTML response
    api_download_failed = False

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
                logging.warning("   Download returned HTML but no confirmation token found.")
                api_download_failed = True

        # Check again in case the retry also returned HTML
        if not api_download_failed and "text/html" in resp.headers.get("Content-Type", "").lower():
            logging.warning("   Download (or retry) returned HTML content.")
            resp.close()
            api_download_failed = True

        if not api_download_failed:
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

            # If we got here, download was successful
            return file_path

    except RequestException as exc:
        logging.error("   Failed to download recording for fallback: %s", exc)
        api_download_failed = True
    except OSError as exc:
        logging.error("   Failed to persist recording for fallback: %s", exc)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError:
            pass
        return None

    # Selenium Fallback (Plan C)
    if api_download_failed and original_url:
        logging.info("   Attempting Selenium fallback for download (Plan C)...")
        try:
            # 1. Navigate to the original URL
            driver.get(original_url)

            # 2. Locate the download button
            # Look for aria-label="Download" or icon="download" or similar
            download_button = None
            try:
                # Wait briefly for page load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
                )

                # Zoom Specific Logic
                if "zoom.us" in original_url:
                    zoom_selectors = [
                        "a.download-key",
                        "[download]",
                        "//a[contains(@href, '/rec/download')]",
                        "//div[@role='button' and contains(., 'Download')]"
                    ]
                    for selector in zoom_selectors:
                        try:
                            if selector.startswith("//"):
                                elements = driver.find_elements(By.XPATH, selector)
                            else:
                                elements = driver.find_elements(By.CSS_SELECTOR, selector)

                            for el in elements:
                                if el.is_displayed():
                                    download_button = el
                                    break
                            if download_button:
                                break
                        except Exception:
                            continue

                # General / Drive Logic (if not found yet)
                if not download_button:
                    # Heuristics for finding download button
                    potential_selectors = [
                        "div[aria-label='Download']",
                        "div[data-tooltip='Download']",
                        "div[role='button'][aria-label='Download']",
                        "div[role='button'] svg[href*='download']", # Not standard SVG usage but sometimes...
                        # Drive specific structure often involves divs acting as buttons
                        "div[role='button'][data-tooltip='Download']",
                    ]

                    for selector in potential_selectors:
                        try:
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                            for el in elements:
                                if el.is_displayed():
                                    download_button = el
                                    break
                            if download_button:
                                break
                        except Exception:
                            continue

                if not download_button:
                    # Try looking for icon name "download" in Material Icons if used?
                    # Or try searching by text?
                    # This part is tricky without specific DOM knowledge, but following instructions:
                    # "Use driver.find_element to locate the download button (look for aria-label="Download" or icon="download")."
                    xpath_selectors = [
                        "//*[@aria-label='Download']",
                        "//*[contains(@aria-label, 'Download')]",
                        "//div[@aria-label='Download']",
                        "//div[text()='Download']"
                    ]
                    for xpath in xpath_selectors:
                         try:
                            elements = driver.find_elements(By.XPATH, xpath)
                            for el in elements:
                                if el.is_displayed():
                                    download_button = el
                                    break
                            if download_button:
                                break
                         except Exception:
                             continue

            except Exception as e:
                logging.debug(f"      Selenium search for download button failed: {e}")

            if download_button:
                logging.info("      Found download button via Selenium. Clicking...")
                driver.execute_script("arguments[0].click();", download_button)

                # 3. Wait for file to appear in download directory
                # We need to watch the downloads directory for a new file.
                # Assuming config.SETTINGS.downloads_dir is where Chrome downloads to.
                # NOTE: Chrome downloads to user's default download dir unless configured otherwise.
                # If the driver was configured to download to config.SETTINGS.downloads_dir, we check there.
                # Otherwise this might be flaky if we don't know where it downloads.
                # Assuming the driver setup configured the download directory.

                # Check for new file appearance
                # Since we don't know the name, we look for the most recent file.

                # Wait loop
                wait_time = 300 # seconds
                start_wait = time.time()
                new_file = None
                last_log_time = start_wait

                # We should probably know the download directory.
                # If `target_dir` is used for requests, does Selenium use it?
                # Usually Selenium driver is configured with a download dir.
                # Let's assume `TEMP_DIR` or `config.SETTINGS.downloads_dir` is the one.
                # The prompt implies we should wait for the file.

                # Since we can't easily change where the browser downloads on the fly without re-init,
                # we rely on the pre-configured download dir.
                # We'll check `config.SETTINGS.downloads_dir` or `target_dir` (which is subdir).

                # Actually, let's try to detect ANY new file in the likely download location.
                # `config.SETTINGS.downloads_dir` seems to be the root.

                monitor_dir = config.SETTINGS.downloads_dir
                initial_files = set(os.listdir(monitor_dir))

                while time.time() - start_wait < wait_time:
                    current_files = set(os.listdir(monitor_dir))
                    new_files = current_files - initial_files

                    # Filter out .crdownload or .tmp files
                    valid_new_files = [f for f in new_files if not f.endswith('.crdownload') and not f.endswith('.tmp')]

                    if valid_new_files:
                        # Grab the first one
                        downloaded_filename = valid_new_files[0]
                        # Move it to our target path `file_path`
                        found_path = os.path.join(monitor_dir, downloaded_filename)

                        # Wait for size to stabilize?
                        # If .crdownload is gone, it should be done.

                        # Verify size
                        if os.path.getsize(found_path) > 10 * 1024:
                            # Move/Rename to our target `file_path`
                            # But `file_path` has an extension based on URL, the download might have different extension.
                            # We should trust the download extension.
                            _, down_ext = os.path.splitext(downloaded_filename)
                            final_path = os.path.splitext(file_path)[0] + down_ext

                            os.rename(found_path, final_path)
                            logging.info(f"      Selenium download successful: {final_path}")
                            return final_path

                    now = time.time()
                    if now - last_log_time >= 30:
                        elapsed = int(now - start_wait)
                        logging.info(f"      Waiting for download... {elapsed} seconds elapsed")
                        last_log_time = now

                    time.sleep(1)

                logging.warning("      Selenium download timed out or file not found.")
            else:
                 logging.warning("      Selenium fallback: Could not locate download button.")

        except Exception as e:
            logging.error(f"      Selenium fallback failed: {e}")

    # Cleanup if everything failed
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass

    return None


def _transcribe_single_chunk(client, file_path: str) -> str:
    """Helper to transcribe a single file/chunk."""
    video_file = None
    try:
        logging.info(f"   Gemini fallback: uploading chunk {os.path.basename(file_path)}...")
        # Upload the file
        video_file = client.files.upload(file=file_path)

        # Wait for processing
        logging.info("   Gemini fallback: waiting for file processing...")
        # Add simple timeout mechanism to avoid infinite loops
        max_retries = 180  # Wait up to 6 minutes for large chunks
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
        logging.error(f"   Gemini chunk transcription failed for {file_path}: {exc}")
        return ""
    finally:
        # Cleanup file on Gemini side
        if video_file:
            try:
                logging.info("   Gemini fallback: deleting remote file %s", video_file.name)
                client.files.delete(name=video_file.name)
            except Exception as e:
                logging.warning("   Failed to delete remote file on Gemini: %s", e)


def _transcribe_with_gemini(file_path: str) -> str:
    if not _can_attempt_gemini_fallback():
        return ""

    logging.info("   Gemini fallback: Checking video duration...")
    client = genai.Client(api_key=_GEMINI_API_KEY)

    temp_chunks = []

    try:
        # Check duration using MoviePy
        duration_sec = 0.0
        try:
            clip = VideoFileClip(file_path)
            duration_sec = clip.duration
            clip.close()
        except Exception as e:
            logging.warning(f"   MoviePy failed to read duration: {e}. Trying ffmpeg...")
            try:
                result = subprocess.run([get_ffmpeg_exe(), "-i", file_path], stderr=subprocess.PIPE, text=True)
                # Parse output
                match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", result.stderr)
                if match:
                    hours, minutes, seconds = map(float, match.groups())
                    duration_sec = hours * 3600 + minutes * 60 + seconds
                else:
                    raise ValueError("Could not find duration in ffmpeg output")
            except Exception as ffmpeg_err:
                logging.error(f"   ffmpeg also failed to determine duration: {ffmpeg_err}")
                logging.error("   Cannot determine duration. Single upload disabled.")
                return ""

        chunk_size_sec = 50 * 60  # 50 minutes

        if duration_sec <= chunk_size_sec:
             return _transcribe_single_chunk(client, file_path)

        logging.info(f"   Video duration ({duration_sec}s) exceeds 50 mins. Chunking...")

        # Split into chunks
        base_name, ext = os.path.splitext(file_path)
        full_transcript = []

        # Re-open clip for sub-clipping (the previous close might have released resources)
        # Using context manager for safety
        with VideoFileClip(file_path) as video:
            num_chunks = int(duration_sec // chunk_size_sec) + 1

            for i in range(num_chunks):
                start_time = i * chunk_size_sec
                end_time = min((i + 1) * chunk_size_sec, duration_sec)

                if start_time >= end_time:
                    break

                chunk_filename = f"{base_name}_chunk_{i}{ext}"
                temp_chunks.append(chunk_filename)

                logging.info(f"   Creating chunk {i+1}/{num_chunks}: {start_time}-{end_time}s")

                # Write chunk to file
                new_clip = video.subclipped(start_time, end_time)
                new_clip.write_videofile(
                    chunk_filename,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile=f"{base_name}_temp_audio.m4a",
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )

                # Upload and transcribe chunk immediately
                chunk_text = _transcribe_single_chunk(client, chunk_filename)
                if chunk_text:
                    full_transcript.append(chunk_text)
                else:
                    logging.warning(f"   Chunk {i+1} failed to transcribe.")

        return "\n".join(full_transcript)

    except Exception as exc:
        logging.error("   Gemini transcription (chunked) failed: %s", exc)
        return ""
    finally:
        # Clean up temp chunks
        for chunk_file in temp_chunks:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                    logging.info(f"   Deleted temp chunk: {chunk_file}")
                except Exception as e:
                    logging.warning(f"   Failed to delete temp chunk {chunk_file}: {e}")


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
    # Pass original URL to _download_recording_media for Plan C fallback
    media_path = _download_recording_media(driver, download_url, original_url=url)
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
