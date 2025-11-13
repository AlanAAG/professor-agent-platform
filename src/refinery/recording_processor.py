import html
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
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

_TIMEDTEXT_URL_REGEX = re.compile(
    r"https://drive\.google\.com[^\"'`<>\s]*timedtext[^\"'`<>\s]*",
    re.IGNORECASE,
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


def extract_transcript(driver: webdriver.Chrome, url: str, resource_type: str) -> str:
    """Public entry point to selectively extract transcript content from a recording URL.

    - For ZOOM_RECORDING: open in a new tab and scrape transcript content.
    - For DRIVE_RECORDING: open in a new tab and retrieve transcript via timedtext endpoint.
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
