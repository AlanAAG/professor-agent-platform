"""
Core Selenium navigation logic for the harvester module.
Handles driver initialization, login, session management, and course navigation.
"""

from __future__ import annotations

import os
import json
import logging
import time
import re
import shutil
import tempfile
import signal
import threading
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple, Optional, Callable, TypeVar, Sequence
from selenium import webdriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
    ElementNotInteractableException,
    InvalidElementStateException,
)
from . import config
from .utils import resilient_find_element

# --- Globals and Initialization ---

# Cache to track which courses have been processed in the current pipeline run.
# Used to prevent navigating to a course link twice if it appears in multiple groups.
_COURSE_LINKS_SEEN_CACHE: set[str] = set()

T = TypeVar("T")

def reset_course_tracking():
    """Resets the seen courses cache for a fresh pipeline run."""
    _COURSE_LINKS_SEEN_CACHE.clear()


_RESOURCE_LINK_SELECTORS: List[str] = [
    "[data-testid*='resource'] a[href]",
    "[data-section*='resource'] a[href]",
    "div.fileBox a[href]",
    "div.fileContentCol a[href]",
    "a[href]",
]

_RESOURCE_LINK_SELECTOR_UNION = ", ".join(dict.fromkeys(_RESOURCE_LINK_SELECTORS))


_JS_LOCATE_SECTION_CONTAINER = """
const header = arguments[0];
const normalizedTitle = (arguments[1] || "").trim().toLowerCase();
if (!header) {
  return null;
}

const isContainer = (el) => {
  if (!el || el.nodeType !== 1) return false;
  if (el.matches("template,script,style")) return false;
  if (el.hasAttribute("data-section-body") || el.hasAttribute("data-section-content")) return true;
  const role = el.getAttribute("role") || "";
  if (role === "region" || role === "tabpanel" || role === "group") return true;
  return Boolean(el.querySelector && el.querySelector("a[href]"));
};

const ariaControls = header.getAttribute && header.getAttribute("aria-controls");
if (ariaControls) {
  const ids = ariaControls.split(/\\s+/).map((s) => s.trim()).filter(Boolean);
  for (const id of ids) {
    const candidate = document.getElementById(id);
    if (isContainer(candidate)) return candidate;
  }
}

let sibling = header.nextElementSibling;
while (sibling) {
  if (isContainer(sibling)) return sibling;
  sibling = sibling.nextElementSibling;
}

let parent = header.parentElement;
while (parent) {
  const next = parent.nextElementSibling;
  if (isContainer(next)) return next;
  parent = parent.parentElement;
}

const labelled = Array.from(document.querySelectorAll("[aria-label], [data-section-title]"));
for (const el of labelled) {
  const label = (el.getAttribute("aria-label") || el.getAttribute("data-section-title") || "")
    .trim()
    .toLowerCase();
  if (label && label === normalizedTitle && isContainer(el)) {
    return el;
  }
}

return null;
"""


def _is_stale(element: WebElement) -> bool:
    try:
        _ = element.is_enabled()
        return False
    except StaleElementReferenceException:
        return True


def _scroll_into_view_center(driver: webdriver.Chrome, element: WebElement) -> None:
    driver.execute_script(
        "arguments[0].scrollIntoView({block: arguments[1], inline: 'nearest'});",
        element,
        "center",
    )


def _wait_for_document_ready(driver: webdriver.Chrome, timeout: float = 1.0) -> None:
    """Wait briefly for the page to report an interactive or complete ready state."""
    try:
        WebDriverWait(driver, timeout, poll_frequency=0.1).until(
            lambda d: (d.execute_script("return document.readyState") or "").lower()
            in {"complete", "interactive"}
        )
    except Exception:
        # Best-effort stabilization; swallow timeouts or script errors.
        pass


def _wait_for_aria_expansion(
    driver: webdriver.Chrome,
    header_supplier: Callable[[webdriver.Chrome], Optional[WebElement]],
    timeout: int = 5,
) -> None:
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: (
                lambda el: False
                if el is None
                else (
                    (el.get_attribute("aria-expanded") or "").strip().lower()
                    in {"", "true", "1"}
                )
            )(header_supplier(d))
        )
    except TimeoutException:
        logging.debug("aria-expanded did not confirm expansion within %ss", timeout)
    except Exception:
        # Non-critical; fallback relies on link waits
        pass


def _make_locator_resolver(locator: Tuple[str, str]) -> Callable[[webdriver.Chrome], Optional[WebElement]]:
    by, value = locator

    def _resolver(driver: webdriver.Chrome) -> Optional[WebElement]:
        try:
            return driver.find_element(by, value)
        except NoSuchElementException:
            return None
        except StaleElementReferenceException:
            return None

    return _resolver


def _resolve_section_container(
    driver: webdriver.Chrome,
    header_el: WebElement,
    header_xpath_used: str,
    section_title: str,
    timeout: int = 10,
) -> Tuple[Optional[WebElement], Optional[Callable[[webdriver.Chrome], Optional[WebElement]]], str, str]:
    strategies: List[Tuple[Tuple[str, str], str]] = []

    def _register_locator(raw_value: str | None, label: str) -> None:
        if not raw_value:
            return
        for token in raw_value.split():
            candidate = token.strip()
            if not candidate:
                continue
            if candidate.startswith("#"):
                strategies.append(((By.CSS_SELECTOR, candidate), label))
            else:
                strategies.append(((By.ID, candidate), label))

    # Inspect header attributes for explicit relationships
    _register_locator(header_el.get_attribute("aria-controls"), "aria-controls")
    for attr in ("data-controls", "data-target", "data-content", "data-section-id", "data-panel-id"):
        raw = header_el.get_attribute(attr)
        if raw and raw.startswith("#"):
            strategies.append(((By.CSS_SELECTOR, raw), attr))
        elif raw:
            strategies.append(((By.ID, raw), attr))

    if header_xpath_used:
        container_xpath_primary = f"{header_xpath_used}/following-sibling::*[self::div or self::section][1]"
        container_xpath_with_links = (
            f"{header_xpath_used}/following-sibling::*[self::div or self::section]"
            "[.//a[@href]][1]"
        )
        strategies.extend([
            ((By.XPATH, container_xpath_primary), "following-sibling"),
            ((By.XPATH, container_xpath_with_links), "following-sibling-links"),
            (
                (By.XPATH,
                 f"{header_xpath_used}/ancestor::*[self::div or self::section][1]/following-sibling::*[self::div or self::section][.//a[@href]][1]"),
                "ancestor-sibling-links",
            ),
        ])

    normalized_title = section_title.strip()
    if normalized_title:
        normalized_xpath = (
            "(//div[.//p[normalize-space(text())='" + normalized_title + "']"
            " or .//h3[normalize-space(text())='" + normalized_title + "']"
            " or .//h4[normalize-space(text())='" + normalized_title + "'])"
            "[1]/following-sibling::*[self::div or self::section][.//a[@href]][1])"
        )
        strategies.append(((By.XPATH, normalized_xpath), "title-proximity"))

    seen_locators: set[Tuple[str, str]] = set()
    for locator, label in list(strategies):
        if locator in seen_locators:
            continue
        seen_locators.add(locator)
        try:
            wait = WebDriverWait(driver, timeout)
            container = wait.until(EC.presence_of_element_located(locator))
            wait.until(EC.visibility_of(container))
            if container:
                descriptor = f"{locator[0]}::{locator[1]}"
                return container, _make_locator_resolver(locator), label, descriptor
        except TimeoutException:
            continue
        except Exception as e:
            logging.debug("Container locator %s failed: %s", label, e)

    # JavaScript proximity fallback
    try:
        header_for_js = header_el
        if _is_stale(header_for_js) and header_xpath_used:
            header_for_js = driver.find_element(By.XPATH, header_xpath_used)
        container_js = driver.execute_script(
            _JS_LOCATE_SECTION_CONTAINER,
            header_for_js,
            section_title,
        )
        if container_js:
            def _resolver(drv: webdriver.Chrome) -> Optional[WebElement]:
                try:
                    header_current = header_for_js
                    if (header_xpath_used and (_is_stale(header_current) or header_current is None)):
                        header_current = drv.find_element(By.XPATH, header_xpath_used)
                except Exception:
                    header_current = None
                if header_current is None:
                    return None
                try:
                    return drv.execute_script(
                        _JS_LOCATE_SECTION_CONTAINER,
                        header_current,
                        section_title,
                    )
                except Exception:
                    return None

            descriptor = "js-proximity"
            return container_js, _resolver, "js-proximity", descriptor
    except Exception as e:
        logging.debug("JavaScript container resolution failed for '%s': %s", section_title, e)

    return None, None, "unresolved", ""


def _wait_for_links_in_container(
    driver: webdriver.Chrome,
    container_supplier: Callable[[webdriver.Chrome], Optional[WebElement]] | None,
    initial_container: Optional[WebElement],
    timeout: int,
) -> List[WebElement]:
    last_container = initial_container

    def _locate_container(drv: webdriver.Chrome) -> Optional[WebElement]:
        nonlocal last_container
        if last_container and not _is_stale(last_container) and last_container.is_displayed():
            return last_container
        if container_supplier:
            refreshed = container_supplier(drv)
            if refreshed is not None:
                last_container = refreshed
                return refreshed
        return last_container

    def _collect(drv: webdriver.Chrome):
        container = _locate_container(drv)
        if not container or _is_stale(container):
            return False
        try:
            anchors = container.find_elements(By.CSS_SELECTOR, _RESOURCE_LINK_SELECTOR_UNION)
        except StaleElementReferenceException:
            return False
        if not anchors:
            return False
        filtered: List[WebElement] = []
        seen_ids: set[str] = set()
        for anchor in anchors:
            try:
                href = (anchor.get_attribute("href") or "").strip()
            except StaleElementReferenceException:
                continue
            if not href:
                continue
            lower_href = href.lower()
            if lower_href.startswith("javascript:") or lower_href == "#":
                continue
            anchor_id = getattr(anchor, "id", None)
            if anchor_id and anchor_id in seen_ids:
                continue
            if anchor_id:
                seen_ids.add(anchor_id)
            filtered.append(anchor)
        return filtered if filtered else False

    try:
        return WebDriverWait(driver, timeout).until(_collect)
    except TimeoutException:
        return []


# --- Driver Management and Session Persistence ---

def _resolve_chrome_binary() -> str | None:
    """Attempt to resolve a Chrome/Chromium binary path from env or PATH.

    Checks common env vars and executable names used across providers
    (e.g., Render, Heroku, Codespaces, Debian/Ubuntu).
    """
    candidate_env_vars = [
        "CHROME_BINARY",
        "GOOGLE_CHROME_SHIM",
        "CHROME_PATH",
        "CHROMIUM_PATH",
    ]
    for env_name in candidate_env_vars:
        binary_path = os.getenv(env_name)
        if binary_path and os.path.exists(binary_path):
            return binary_path

    # Fallback: search common executable names
    for exe in ("google-chrome", "chrome", "chromium", "chromium-browser"):
        resolved = shutil.which(exe)
        if resolved:
            return resolved
    return None


def _ensure_tmp_dirs() -> dict[str, str]:
    """Ensure Chrome temp directories exist and return their paths.

    Using tmp directories improves stability in containerized environments
    by avoiding permission errors in default locations.
    """
    base = Path("/tmp/harvester_chrome")
    user_data = base / "user_data"
    data_path = base / "data"
    cache_dir = base / "cache"
    for p in (user_data, data_path, cache_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "user_data_dir": str(user_data),
        "data_path": str(data_path),
        "disk_cache_dir": str(cache_dir),
    }


def _get_chrome_options(*, force_legacy_headless: bool = False) -> webdriver.ChromeOptions:
    """Configures Chrome options based on environment settings.

    force_legacy_headless: when True, use the legacy "--headless" flag instead of
    the modern "--headless=new". Useful as a fallback for older Chrome builds.
    """
    options = webdriver.ChromeOptions()

    if config.SETTINGS.selenium_headless:
        if force_legacy_headless:
            options.add_argument("--headless")
        else:
            # Use the modern, more stable headless mode (supported by CfT/modern Chrome)
            options.add_argument("--headless=new")

    # Core stability flags for Linux containers
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    # Allow Chrome's DevTools pipe to bind reliably in headless=new mode
    options.add_argument("--remote-debugging-port=0")

    # Use dedicated temp directories to avoid permission issues
    tmp_dirs = _ensure_tmp_dirs()
    options.add_argument(f"--user-data-dir={tmp_dirs['user_data_dir']}")
    options.add_argument(f"--data-path={tmp_dirs['data_path']}")
    options.add_argument(f"--disk-cache-dir={tmp_dirs['disk_cache_dir']}")

    # Tries to mitigate bot detection/fingerprinting
    options.add_experimental_option("excludeSwitches", ["enable-automation"]) \
        if hasattr(options, "add_experimental_option") else None
    options.add_experimental_option("useAutomationExtension", False) \
        if hasattr(options, "add_experimental_option") else None

    # If a Chrome binary is available in the environment, use it explicitly
    chrome_binary = _resolve_chrome_binary()
    if chrome_binary:
        options.binary_location = chrome_binary

    return options


def _create_driver(download_path: str | None = None) -> webdriver.Chrome:
    """Initializes and returns a configured Chrome WebDriver with fallbacks.

    Strategy:
      1) Try modern headless mode with Selenium Manager.
      2) On failure, retry with legacy headless flag.
      3) Provide actionable error logs and hints if both attempts fail.
    """

    def _apply_download_prefs(opts: webdriver.ChromeOptions) -> None:
        if download_path:
            prefs = {
                "download.default_directory": download_path,
                "download.prompt_for_download": False,
            }
            try:
                opts.add_experimental_option("prefs", prefs)
            except Exception:
                # Non-fatal if experimental options aren't supported in this context
                pass

    # Prepare service with log output for easier debugging
    log_path = os.getenv("SELENIUM_LOG_PATH", str(Path.cwd() / "selenium_driver.log"))
    try:
        service = ChromeService(log_output=log_path)
    except TypeError:
        # Older Selenium may not support log_output kwarg; fall back silently
        service = ChromeService()

    # Attempt 1: modern headless
    options = _get_chrome_options(force_legacy_headless=False)
    _apply_download_prefs(options)
    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(config.SETTINGS.page_load_timeout)
        logging.info("WebDriver initialized (headless=new) using Selenium Manager.")
        return driver
    except Exception as first_error:
        logging.warning(
            "Primary WebDriver init failed with headless=new. Retrying with legacy headless. Error: %s",
            first_error,
        )

    # Attempt 2: legacy headless (better compatibility on some older Chrome builds)
    legacy_options = _get_chrome_options(force_legacy_headless=True)
    _apply_download_prefs(legacy_options)
    try:
        driver = webdriver.Chrome(service=service, options=legacy_options)
        driver.set_page_load_timeout(config.SETTINGS.page_load_timeout)
        logging.info("WebDriver initialized (legacy --headless) using Selenium Manager.")
        return driver
    except Exception as second_error:
        # Provide helpful guidance for likely root causes.
        logging.critical(
            "Failed to initialize WebDriver after fallbacks. This often means Chrome/Chromium is missing or "
            "required OS libraries are not installed (e.g., libnss3, libgbm, libasound2, libx11). "
            "If running in a container/CI, install a modern Chrome/Chromium and common runtime libraries. "
            "Also see %s for ChromeDriver logs. Error: %s",
            log_path,
            second_error,
        )
        raise


def _force_kill_driver_process(driver: webdriver.Chrome) -> None:
    """Best-effort termination of lingering Chrome/Chromedriver processes."""
    service = getattr(driver, "service", None)
    process = getattr(service, "process", None)
    pid = getattr(process, "pid", None)

    if not process:
        logging.debug("No WebDriver service process found to terminate.")
        return

    if pid:
        logging.warning("Attempting to forcefully terminate lingering WebDriver process (pid=%s).", pid)
    else:
        logging.warning("Attempting to forcefully terminate lingering WebDriver process with unknown pid.")

    try:
        process.terminate()
    except Exception as terminate_err:
        logging.debug("process.terminate() failed: %s", terminate_err)

    try:
        process.wait(timeout=3)
        logging.info("WebDriver process terminated after forced kill attempt.")
        return
    except Exception as wait_err:
        logging.debug("Waiting for WebDriver process termination failed: %s", wait_err)

    try:
        process.kill()
        logging.info("WebDriver process killed via process.kill().")
    except Exception as kill_err:
        logging.debug("process.kill() failed: %s", kill_err)

    if pid:
        sig = signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM
        try:
            os.kill(pid, sig)
            logging.info("WebDriver OS-level kill signal dispatched for pid %s.", pid)
        except Exception as os_kill_err:
            logging.debug("OS-level kill attempt for pid %s failed: %s", pid, os_kill_err)

def _load_session_state(driver: webdriver.Chrome):
    """Loads cookies and localStorage from the persisted auth state file."""
    if not os.path.exists(config.AUTH_STATE_FILE):
        return
    
    try:
        with open(config.AUTH_STATE_FILE, "r") as f:
            state: Dict[str, Any] = json.load(f)
        
        # Load cookies
        cookies = state.get("cookies", [])
        if cookies:
            # Must navigate to the domain first to set cookies
            driver.get(config.BASE_URL)
            for cookie in cookies:
                # Selenium requires "expiry" to be an integer timestamp
                if "expiry" in cookie and isinstance(cookie["expiry"], float):
                    cookie["expiry"] = int(cookie["expiry"])
                # Only add necessary keys
                if cookie.get("domain") and cookie.get("name") and cookie.get("value"):
                    driver.add_cookie(cookie)
            logging.info(f"Loaded {len(cookies)} cookies.")
        
        # Load localStorage (less common, but useful for some apps)
        local_storage_data = state.get("localStorage", {})
        if local_storage_data:
            # Execute script to restore localStorage items
            for key, value in local_storage_data.items():
                driver.execute_script(f"window.localStorage.setItem(arguments[0], arguments[1]);", key, value)
            logging.info(f"Loaded {len(local_storage_data)} localStorage items.")

    except Exception as e:
        logging.warning(f"Failed to load session state: {e}")


def _save_session_state(driver: webdriver.Chrome):
    """Saves cookies and localStorage to the persisted auth state file."""
    try:
        state = {
            "cookies": driver.get_cookies(),
            "localStorage": driver.execute_script("return window.localStorage;"),
            "timestamp": time.time(),
        }
        os.makedirs(os.path.dirname(config.AUTH_STATE_FILE), exist_ok=True)
        with open(config.AUTH_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
        logging.info(f"Session state saved to {config.AUTH_STATE_FILE}.")
    except Exception as e:
        logging.warning(f"Failed to save session state: {e}")


def _take_error_screenshot(driver: webdriver.Chrome, filename_prefix: str):
    """Saves a screenshot to the configured directory with a timestamp."""
    try:
        # Create a unique filename with timestamp and prefix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.png"
        filepath = os.path.join(config.SETTINGS.screenshot_dir, filename)
        
        # Ensure the directory exists
        os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
        
        # Save the screenshot
        driver.save_screenshot(filepath)
        logging.critical(f"CRITICAL ERROR: Screenshot saved to {filepath}")
    except Exception as e:
        # Avoid causing a new error during error logging
        logging.error(f"Failed to capture error screenshot: {e}")


def is_session_valid(driver: webdriver.Chrome) -> bool:
    """Checks if the current session state is likely still logged in."""
    try:
        driver.get(config.COURSES_URL)
        # Check if we were redirected back to the login page
        if driver.current_url.startswith(config.LOGIN_URL):
            logging.info("Session invalid: Redirected to login page.")
            return False
        
        # Check for a reliable dashboard element
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, config.DASHBOARD_INDICATOR_CSS))
        )
        logging.info("Session valid: Dashboard indicator found.")
        return True
    except TimeoutException:
        logging.info("Session status inconclusive (timeout).")
        return False
    except Exception as e:
        logging.warning(f"Error checking session validity: {e}")
        return False


def perform_login(driver: webdriver.Chrome) -> bool:
    """Performs the login sequence using stored credentials, with retry logic."""
    username = os.getenv("COACH_USERNAME")
    password = os.getenv("COACH_PASSWORD")
    if not username or not password:
        logging.critical("COACH_USERNAME or COACH_PASSWORD environment variable not set.")
        return False

    MAX_LOGIN_ATTEMPTS = 3

    for attempt in range(1, MAX_LOGIN_ATTEMPTS + 1):
        try:
            logging.info(f"Attempting fresh login (Attempt {attempt}/{MAX_LOGIN_ATTEMPTS})...")

            # 1. Navigate (or re-navigate) to Login Page
            driver.get(config.LOGIN_URL)
            _wait_for_document_ready(driver, timeout=2.0)

            # --- START CRITICAL RELOAD CHECK ---
            try:
                # Check for the primary username input field. If it's missing, the page likely failed to load.
                # NOTE: EC.presence_of_element_located requires a tuple (By, selector_value).
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located(config.USERNAME_SELECTORS[0])
                )
            except TimeoutException:
                # Page failed to load critical element (e.g., received "Failed to load page" error).
                logging.warning("Login page failed to load critical element. Forcing page refresh for recovery.")
                driver.refresh()
                _wait_for_document_ready(driver, timeout=2.0)
            # --- END CRITICAL RELOAD CHECK ---

            # 2. Enter credentials after ensuring interactable inputs
            _resilient_type(
                driver,
                config.USERNAME_SELECTORS,
                username,
                "username field",
                timeout=config.SETTINGS.wait_timeout,
            )
            _resilient_type(
                driver,
                config.PASSWORD_SELECTORS,
                password,
                "password field",
                timeout=config.SETTINGS.wait_timeout,
            )

            # 3. Click login button after explicit wait
            _resilient_click(
                driver,
                config.LOGIN_BUTTON_SELECTORS,
                "login button",
                timeout=config.SETTINGS.wait_timeout,
            )

            # --- Use an extended wait for post-login redirection ---
            EXTENDED_WAIT_TIMEOUT = 45

            # 4. Wait for redirection to the dashboard (or timeout)
            WebDriverWait(driver, EXTENDED_WAIT_TIMEOUT).until(
                EC.url_contains(config.COURSES_URL)
            )

            logging.info(f"Login successful after {attempt} attempt(s).")
            _save_session_state(driver)
            return True

        except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
            logging.warning(f"Login attempt {attempt} failed (Element/Timeout): {e}. Retrying.")
            _wait_for_document_ready(driver, timeout=1.0)
            continue

        except Exception as e:
            if "Timed out receiving message from renderer" in str(e):
                logging.warning(
                    f"Login attempt {attempt} failed (Renderer Timeout). Pausing and retrying."
                )
                _wait_for_document_ready(driver, timeout=2.5)
                continue
            else:
                logging.error(f"Login failed on attempt {attempt} due to critical error: {e}")
                break

    logging.error(f"Login failed after {MAX_LOGIN_ATTEMPTS} attempts.")
    try:
        os.makedirs(config.SETTINGS.screenshot_dir, exist_ok=True)
    except Exception as mkdir_err:
        logging.warning("Unable to create screenshot directory '%s': %s", config.SETTINGS.screenshot_dir, mkdir_err)
    driver.save_screenshot(os.path.join(config.SETTINGS.screenshot_dir, "login_failure.png"))
    return False


def _on_login_page(driver: webdriver.Chrome) -> bool:
    login_url = getattr(config, "LOGIN_URL", "")
    if not login_url:
        return False
    try:
        current_url = driver.current_url or ""
    except Exception:
        return False
    normalized_login = login_url.rstrip("/")
    normalized_current = current_url.rstrip("/")
    return normalized_current.startswith(normalized_login)


def _exception_indicates_session_timeout(exc: Exception) -> bool:
    if isinstance(exc, TimeoutException):
        return True
    if isinstance(exc, WebDriverException):
        message = str(exc).lower()
        timeout_markers = (
            "timeout",
            "session not found",
            "invalid session id",
            "chrome not reachable",
            "disconnected",
        )
        return any(marker in message for marker in timeout_markers)
    return False


def _session_timeout_detected(driver: webdriver.Chrome, exc: Optional[Exception]) -> bool:
    if _on_login_page(driver):
        return True
    if exc is not None and _exception_indicates_session_timeout(exc):
        return True
    return False


def resilient_session_action(
    driver: webdriver.Chrome,
    action_func: Callable[..., T],
    *args,
    **kwargs,
) -> T:
    """Execute an action with automatic session re-authentication on timeout."""

    action_label = getattr(action_func, "__name__", repr(action_func))

    for attempt in range(1, 3):
        try:
            result = action_func(driver, *args, **kwargs)
        except Exception as exc:
            if _session_timeout_detected(driver, exc):
                if attempt == 1:
                    logging.warning(
                        "Session expired while executing %s; attempting re-authentication.",
                        action_label,
                    )
                    if not perform_login(driver):
                        raise RuntimeError(
                            f"Re-authentication failed while executing {action_label}."
                        ) from exc
                    continue
                raise
            raise
        else:
            if _session_timeout_detected(driver, None):
                if attempt == 1:
                    logging.warning(
                        "Session expired while executing %s; attempting re-authentication.",
                        action_label,
                    )
                    if not perform_login(driver):
                        raise RuntimeError(
                            f"Re-authentication failed while executing {action_label}."
                        )
                    continue
                raise RuntimeError(f"{action_label} redirected to login page after retry.")
            return result

    raise RuntimeError(f"{action_label} failed after re-authentication attempt.")


@contextmanager
def launch_and_login() -> webdriver.Chrome:
    """Context manager to initialize driver, manage session, and ensure login."""
    driver = _create_driver(download_path=config.SETTINGS.downloads_dir)
    try:
        _load_session_state(driver)
        
        if not is_session_valid(driver):
            logging.info("Session invalid or expired. Attempting fresh login.")
            if not perform_login(driver):
                raise RuntimeError("Harvester failed to log in.")
        
        yield driver
        
    finally:
        if driver:
            timeout_seconds = 5
            quit_completed = threading.Event()
            # Ensure quit is attempted even if driver is in invalid state
            if hasattr(driver, "quit"):
                def _on_quit_timeout():
                    if quit_completed.is_set():
                        return
                    logging.warning(
                        "WebDriver quit timed out after %s seconds; attempting forced termination.",
                        timeout_seconds,
                    )
                    _force_kill_driver_process(driver)

                timer = threading.Timer(timeout_seconds, _on_quit_timeout)
                timer.daemon = True
                timer.start()
                try:
                    try:
                        driver.quit()
                        quit_completed.set()
                        logging.info("WebDriver quit successfully.")
                    except Exception as e:
                        quit_completed.set()
                        # Swallow all exceptions - quit() may fail if browser crashed
                        logging.warning(f"WebDriver quit failed (browser may have crashed): {e}")
                        # Don't re-raise - we're in a finally block and cleanup is best-effort
                        _force_kill_driver_process(driver)
                finally:
                    timer.cancel()
            else:
                logging.debug("Driver object missing 'quit'; skipping cleanup.")


# --- Core Navigation Helpers ---

def _wait_for_resilient_element(
    driver: webdriver.Chrome,
    selectors: Sequence[Tuple[str, str]],
    name: str,
    *,
    timeout: int,
    require_clickable: bool = False,
) -> WebElement:
    """Wait for an element via resilient selectors, ensuring visibility/interactability."""

    wait = WebDriverWait(
        driver,
        timeout,
        ignored_exceptions=(StaleElementReferenceException,),
    )

    def _locate(drv: webdriver.Chrome):
        try:
            element = resilient_find_element(drv, selectors, name)
        except NoSuchElementException:
            return False
        except StaleElementReferenceException:
            return False

        try:
            if require_clickable:
                return element if element.is_displayed() and element.is_enabled() else False
            return element if element.is_displayed() else False
        except StaleElementReferenceException:
            return False

    return wait.until(_locate)


def _resilient_type(
    driver: webdriver.Chrome,
    selectors: Sequence[Tuple[str, str]],
    text: str,
    name: str,
    *,
    timeout: int,
    clear: bool = True,
    click_before: bool = True,
) -> WebElement:
    """Type into an element resolved via resilient selectors with retry logic."""

    last_error: Optional[Exception] = None

    for attempt in range(1, 4):
        try:
            element = _wait_for_resilient_element(
                driver,
                selectors,
                name,
                timeout=timeout,
                require_clickable=True,
            )

            try:
                _scroll_into_view_center(driver, element)
            except Exception:
                pass

            if click_before:
                try:
                    element.click()
                except Exception:
                    try:
                        driver.execute_script("arguments[0].focus();", element)
                    except Exception:
                        pass

            if clear:
                try:
                    element.clear()
                except StaleElementReferenceException:
                    raise

            element.send_keys(text)
            logging.debug("Typed into %s using resilient selectors on attempt %s", name, attempt)
            return element
        except TimeoutException as err:
            last_error = err
            break
        except (StaleElementReferenceException, ElementNotInteractableException, InvalidElementStateException) as err:
            last_error = err
            logging.debug("Retrying resilient type for %s due to %s (attempt %s)", name, err, attempt)
            time.sleep(0.3)

    if isinstance(last_error, TimeoutException):
        raise last_error

    raise TimeoutException(
        f"Unable to interact with {name} after multiple resilient selector attempts. Last error: {last_error}"
    ) from last_error


def _resilient_click(
    driver: webdriver.Chrome,
    selectors: Sequence[Tuple[str, str]],
    name: str,
    *,
    timeout: int,
) -> WebElement:
    """Click an element resolved via resilient selectors using JS for reliability."""

    last_error: Optional[Exception] = None

    for attempt in range(1, 4):
        try:
            element = _wait_for_resilient_element(
                driver,
                selectors,
                name,
                timeout=timeout,
                require_clickable=True,
            )

            try:
                _scroll_into_view_center(driver, element)
            except Exception:
                pass

            driver.execute_script("arguments[0].click();", element)
            logging.info("Clicked %s using resilient selectors (attempt %s)", name, attempt)
            return element
        except TimeoutException as err:
            last_error = err
            break
        except (StaleElementReferenceException, ElementNotInteractableException) as err:
            last_error = err
            logging.debug("Retrying resilient click for %s due to %s (attempt %s)", name, err, attempt)
            time.sleep(0.2)

    if isinstance(last_error, TimeoutException):
        raise last_error

    raise TimeoutException(
        f"Unable to click {name} after multiple resilient selector attempts. Last error: {last_error}"
    ) from last_error


def safe_find(
    driver: webdriver.Chrome,
    locator: Tuple[str, str],
    timeout: int = 30,
    clickable: bool = False,
) -> WebElement:
    """Robust element finding with explicit waits."""
    wait = WebDriverWait(
        driver,
        timeout,
        ignored_exceptions=(StaleElementReferenceException,),
    )
    condition = EC.element_to_be_clickable(locator) if clickable else EC.presence_of_element_located(locator)

    try:
        return wait.until(condition)
    except (TimeoutException, NoSuchElementException) as e:
        # --- START CRITICAL ERROR LOGGING ---
        locator_string = f"{locator[0]}_{locator[1].replace('//', '').replace('/', '_').replace('[', '_').replace(']', '')[:50]}"
        _take_error_screenshot(driver, f"find_timeout_{locator_string}")
        # --- END CRITICAL ERROR LOGGING ---
        raise TimeoutException(f"Timed out waiting for element located by {locator}. Original Error: {e}")


def safe_find_all(
    driver: webdriver.Chrome,
    locator: Tuple[str, str],
    timeout: int = 30,
) -> List[WebElement]:
    """Robust find-all operation with explicit wait. Returns empty list on failure."""
    try:
        wait = WebDriverWait(driver, timeout)
        wait.until(EC.presence_of_element_located(locator))
        return driver.find_elements(*locator)
    except TimeoutException:
        logging.debug(f"No elements found for {locator} within {timeout}s")
        return []
    except Exception as e:
        logging.warning(f"Error finding elements {locator}: {e}")
        return []


def safe_click(driver: webdriver.Chrome, locator: Tuple[str, str], timeout: int = 30):
    """Robust click operation using JavaScript to bypass potential overlays/stale elements."""
    element = safe_find(driver, locator, timeout, clickable=True)
    # Use JavaScript click (often more reliable than native .click() in headless environments)
    driver.execute_script("arguments[0].click();", element)
    logging.info(f"Clicked element located by {locator}")


def safe_type(
    driver: webdriver.Chrome,
    locator: Tuple[str, str],
    text: str,
    *,
    timeout: int = 30,
    clear: bool = True,
    click_before: bool = True,
):
    """Type text into an input after waiting for it to be interactable."""
    wait = WebDriverWait(
        driver,
        timeout,
        poll_frequency=0.2,
        ignored_exceptions=(StaleElementReferenceException,),
    )
    last_error: Optional[Exception] = None

    for attempt in range(1, 4):
        try:
            element = wait.until(EC.element_to_be_clickable(locator))
            try:
                _scroll_into_view_center(driver, element)
            except Exception:
                pass

            if click_before:
                try:
                    element.click()
                except Exception:
                    try:
                        driver.execute_script("arguments[0].focus();", element)
                    except Exception:
                        pass

            if clear:
                try:
                    element.clear()
                except StaleElementReferenceException:
                    logging.debug("Input %s became stale during clear; retrying.", locator)
                    raise

            element.send_keys(text)
            logging.debug("Typed into element located by %s on attempt %s", locator, attempt)
            return element
        except TimeoutException as err:
            last_error = err
            logging.debug(
                "Timeout waiting for interactable element %s on attempt %s: %s",
                locator,
                attempt,
                err,
            )
            break
        except (StaleElementReferenceException, ElementNotInteractableException, InvalidElementStateException) as err:
            last_error = err
            logging.debug(
                "Retrying typing into %s due to %s (attempt %s)",
                locator,
                err,
                attempt,
            )
            time.sleep(0.3)

    if isinstance(last_error, TimeoutException):
        raise last_error

    raise TimeoutException(
        f"Unable to interact with element located by {locator} after multiple attempts. Last error: {last_error}"
    ) from last_error


def _get_available_course_codes_impl(driver: webdriver.Chrome) -> set[str]:
    try:
        driver.get(config.COURSES_URL)
        WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
            EC.presence_of_element_located((By.XPATH, config.GROUP_HEADER_XPATH_TEMPLATE.format(group_name=".")))
        )
        # Find all anchor tags that contain courseCode in the href
        course_links = driver.find_elements(By.XPATH, "//a[contains(@href, 'courseCode=')]")
        codes: set[str] = set()
        for link in course_links:
            href = link.get_attribute("href")
            if href:
                # Extract courseCode={code} from the URL
                code_match = re.search(r'courseCode=([^&]+)', href)
                if code_match:
                    codes.add(code_match.group(1))
        return codes
    except Exception as e:
        logging.warning(f"Could not retrieve available course codes: {e}")
        return set()


def get_available_course_codes(driver: webdriver.Chrome) -> set[str]:
    """Scrapes the course dashboard for all course codes visible to the user."""
    return resilient_session_action(driver, _get_available_course_codes_impl)


def _find_and_click_course_link_impl(
    driver: webdriver.Chrome,
    course_code: str,
    group_name: str | None = None,
):
    # 1) Avoid duplicate navigations within a single pipeline run
    if course_code in _COURSE_LINKS_SEEN_CACHE:
        logging.warning(
            "Course %s already processed in this run. Skipping navigation to avoid duplication.",
            course_code,
        )
        return

    # 2) Load the courses page fresh
    driver.get(config.COURSES_URL)

    # 3) Decide path: default-visible vs. grouped
    is_default_visible = False
    try:
        is_default_visible = course_code in getattr(config, "DEFAULT_VISIBLE_COURSES", set())
    except Exception:
        is_default_visible = False

    course_locator = (By.XPATH, config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=course_code))

    try:
        if is_default_visible:
            # PATH 1: Directly click the course link
            logging.info("Course %s is default-visible. Clicking link directly...", course_code)
            course_link = safe_find(
                driver,
                course_locator,
                timeout=config.SETTINGS.wait_timeout,
                clickable=True,
            )
            _scroll_into_view_center(driver, course_link)
            course_link = WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
                EC.element_to_be_clickable(course_locator)
            )
            driver.execute_script("arguments[0].click();", course_link)
        else:
            # PATH 2: Expand the group first, then click the course link
            resolved_group = group_name
            if not resolved_group:
                # Attempt to resolve from COURSE_MAP if not provided
                try:
                    resolved_group = config.COURSE_MAP.get(course_code, {}).get("group")
                except Exception:
                    resolved_group = None

            if not resolved_group:
                logging.warning(
                    "Group for course %s not defined. Attempting page-wide search for course link.",
                    course_code,
                )
            else:
                logging.info("Expanding group '%s' for course %s...", resolved_group, course_code)
                group_locator = (
                    By.XPATH,
                    config.GROUP_HEADER_XPATH_TEMPLATE.format(group_name=resolved_group),
                )
                try:
                    group_header = safe_find(driver, group_locator, timeout=15, clickable=True)
                    # Scroll and click to expand
                    _scroll_into_view_center(driver, group_header)
                    group_header = WebDriverWait(driver, 15).until(
                        EC.element_to_be_clickable(group_locator)
                    )
                    driver.execute_script("arguments[0].click();", group_header)
                    logging.info("Clicked group header '%s'", resolved_group)
                except TimeoutException:
                    logging.warning(
                        "Group header '%s' not found or not clickable. Proceeding with full-page search.",
                        resolved_group,
                    )

            # Now find and click the course link
            course_link = safe_find(
                driver,
                course_locator,
                timeout=config.SETTINGS.wait_timeout,
                clickable=True,
            )
            _scroll_into_view_center(driver, course_link)
            course_link = WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
                EC.element_to_be_clickable(course_locator)
            )
            driver.execute_script("arguments[0].click();", course_link)

        # 4) Confirm navigation reached the course page
        WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
            EC.url_contains(f"courseCode={course_code}")
        )
        _COURSE_LINKS_SEEN_CACHE.add(course_code)
        logging.info("Successfully navigated to course: %s", course_code)

    except TimeoutException as e:
        logging.error("Course link not found or navigation timed out for %s: %s", course_code, e)
        _take_error_screenshot(driver, f"nav_timeout_{course_code}")
        raise
    except Exception as e:
        logging.error("Error clicking course link for %s: %s", course_code, e)
        _take_error_screenshot(driver, f"nav_error_{course_code}")
        raise


def find_and_click_course_link(
    driver: webdriver.Chrome,
    course_code: str,
    group_name: str | None = None,
):
    """Navigate from the courses dashboard into a specific course page.

    Logic mirrors the stable Colab flow:
      - If the course is among default-visible cards, click it directly.
      - Otherwise, expand the owning group header first, then click the link.

    Uses JS-based clicks and scroll-into-view to maximize reliability.
    """

    return resilient_session_action(
        driver,
        _find_and_click_course_link_impl,
        course_code,
        group_name=group_name,
    )


def _navigate_to_resources_section_impl(driver: webdriver.Chrome) -> bool:
    normalized_resources_xpath = (
        "translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='resources'"
    )

    locator_candidates: List[Tuple[str, str]] = [
        (By.XPATH, config.RESOURCES_TAB_XPATH),
        (By.CSS_SELECTOR, "[data-testid='resources-tab']"),
        (By.CSS_SELECTOR, "[data-section='resources']"),
        (By.XPATH, f"//button[{normalized_resources_xpath}]"),
        (By.XPATH, f"//*[(@role='tab' or @role='button') and {normalized_resources_xpath}]")
    ]

    last_error: Optional[Exception] = None

    def _tab_supplier(by: str, value: str) -> Callable[[webdriver.Chrome], Optional[WebElement]]:
        def _supply(drv: webdriver.Chrome) -> Optional[WebElement]:
            try:
                return drv.find_element(by, value)
            except (NoSuchElementException, StaleElementReferenceException):
                return None
        return _supply

    def _tab_activated(tab: Optional[WebElement]) -> bool:
        if not tab or _is_stale(tab):
            return False
        attributes = [
            (tab.get_attribute(attr) or "").strip().lower()
            for attr in ("aria-selected", "aria-current", "data-selected", "data-active")
        ]
        if any(val in {"true", "1", "current", "active"} for val in attributes):
            return True
        class_attr = (tab.get_attribute("class") or "").lower()
        return any(flag in class_attr for flag in ("active", "selected"))

    css_probe = ", ".join([
        "[data-section-title]",
        "[data-testid*='resource-section']",
        "div.fileBox",
    ])

    def _resources_loaded(drv: webdriver.Chrome, tab_getter: Callable[[webdriver.Chrome], Optional[WebElement]]) -> bool:
        tab_candidate = tab_getter(drv)
        if _tab_activated(tab_candidate):
            return True
        # Fallback: check for resource containers appearing in DOM
        return bool(drv.find_elements(By.CSS_SELECTOR, css_probe))

    for by, value in locator_candidates:
        tab_getter = _tab_supplier(by, value)
        try:
            wait = WebDriverWait(
                driver,
                timeout=10,
                ignored_exceptions=(StaleElementReferenceException, NoSuchElementException),
            )
            tab = wait.until(EC.element_to_be_clickable((by, value)))
            _scroll_into_view_center(driver, tab)
            driver.execute_script("arguments[0].click();", tab)
            time.sleep(1.0)
            WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
                lambda drv, supplier=tab_getter: _resources_loaded(drv, supplier)
            )
            logging.info("Navigated to Resources section via locator (%s, %s)", by, value)
            return True
        except Exception as exc:
            last_error = exc
            logging.debug("Resources tab activation failed for locator (%s, %s): %s", by, value, exc)
            continue

    logging.warning(
        "Resources tab not found or failed to activate using available locators. Last error: %s",
        last_error,
    )
    if last_error:
        _take_error_screenshot(driver, "resources_tab_final_fail")
    return False


def navigate_to_resources_section(driver: webdriver.Chrome) -> bool:
    """Click the "Resources" tab on the course page, with selector fallbacks."""

    return resilient_session_action(driver, _navigate_to_resources_section_impl)


def _expand_section_and_get_items_impl(driver: webdriver.Chrome, section_title: str) -> Tuple[str, List[WebElement]]:

    wait_timeout = max(10, min(25, config.SETTINGS.wait_timeout))

    try:
        header_el, header_xpath_used, strategy = _locate_section_header_robust(
            driver,
            section_title,
            timeout=wait_timeout // 2,
        )
        logging.debug(
            "Section header '%s' resolved via strategy '%s' (xpath=%s)",
            section_title,
            strategy,
            header_xpath_used,
        )

        def _header_supplier(drv: webdriver.Chrome) -> Optional[WebElement]:
            nonlocal header_el
            if header_el and not _is_stale(header_el):
                return header_el
            if header_xpath_used:
                try:
                    header_el = drv.find_element(By.XPATH, header_xpath_used)
                except NoSuchElementException:
                    header_el = None
            return header_el

        fresh_header = _header_supplier(driver)
        if not fresh_header:
            logging.info("Header supplier returned None while expanding '%s'", section_title)
            return "", []

        _scroll_into_view_center(driver, fresh_header)

        should_click = True
        try:
            aria_state = (fresh_header.get_attribute("aria-expanded") or "").strip().lower()
            if aria_state in {"true", "1"}:
                should_click = False
        except Exception:
            pass

        if should_click:
            try:
                driver.execute_script("arguments[0].click();", fresh_header)
            except StaleElementReferenceException:
                refreshed = _header_supplier(driver)
                if refreshed:
                    driver.execute_script("arguments[0].click();", refreshed)
                    header_el = refreshed
            except Exception as click_err:
                logging.error("Failed to click section header '%s': %s", section_title, click_err)
                raise
            else:
                logging.info("Clicked section header '%s' via strategy '%s'", section_title, strategy)

        _wait_for_aria_expansion(driver, _header_supplier, timeout=6)

        container_el, container_resolver, container_strategy, container_descriptor = _resolve_section_container(
            driver,
            header_el or fresh_header,
            header_xpath_used,
            section_title,
            timeout=wait_timeout // 2,
        )

        if not container_el:
            logging.info(
                "Unable to resolve container for section '%s' (strategy=%s)",
                section_title,
                container_strategy,
            )
            return container_descriptor, []

        anchors = _wait_for_links_in_container(
            driver,
            container_resolver,
            container_el,
            timeout=wait_timeout,
        )

        if not anchors:
            logging.info(
                "Section '%s' resolved (strategy=%s) but produced no anchors within %ss",
                section_title,
                container_strategy,
                wait_timeout,
            )
            return container_descriptor, []

        logging.info(
            "Extracted %d anchor(s) for section '%s' using container strategy '%s'",
            len(anchors),
            section_title,
            container_strategy,
        )
        return container_descriptor, anchors

    except TimeoutException:
        logging.info("Section '%s' contains no items or timed out during expansion.", section_title)
        return "", []
    except Exception as e:
        logging.error("Error expanding section '%s': %s", section_title, e)
        raise

def expand_section_and_get_items(driver: webdriver.Chrome, section_title: str) -> Tuple[str, List[WebElement]]:
    """
    Expand a resource section and return anchor elements discovered within its container.

    The routine avoids brittle structural XPath assumptions by:
      * Locating headers through multi-strategy fallbacks (_locate_section_header_robust)
      * Resolving containers through attribute inspection, structural heuristics, and JS proximity
      * Waiting explicitly for populated anchors instead of relying on fixed sleeps
    """

    return resilient_session_action(
        driver,
        _expand_section_and_get_items_impl,
        section_title,
    )


# --- Robust replacements (appended to ensure override of earlier definitions) ---


def _locate_section_header_robust(
    driver: webdriver.Chrome,
    section_title: str,
    timeout: int = 5,
) -> Tuple[WebElement, str, str]:
    """Locate the section header element robustly with multi-level fallbacks.

    Strategy (in order):
    - primary: Use configured XPath template targeting the header container with exact title match.
    - fallback_1: Find a div that contains a <p> with exact text and a 'name' class (header-like).
    - fallback_2: Find any element on the page whose visible text exactly matches the title,
      then resolve to its nearest ancestor <div> as the clickable header.

    Returns a tuple of (header_element, header_xpath_used, strategy_label).
    """
    # Attempt 1: Primary config XPath (exact title within stable header container)
    primary_xpath = config.SECTION_HEADER_XPATH_TPL.format(section_title=section_title)
    try:
        locator = (By.XPATH, primary_xpath)
        header = safe_find(driver, locator, timeout=timeout)
        WebDriverWait(driver, timeout).until(EC.visibility_of(header))
        # Ensure clickable
        WebDriverWait(driver, timeout).until(lambda d: header.is_displayed() and header.is_enabled())
        logging.debug("Section header located via primary XPath: %s", primary_xpath)
        return header, primary_xpath, "primary"
    except (TimeoutException, NoSuchElementException) as e:
        logging.info("Primary header locator failed for '%s': %s", section_title, e)

    # Attempt 2: Header-like div that contains a <p name> with the exact title
    # Targets structures like: <div ...><div class='leftHeader'><p class='name'>Title</p></div></div>
    fb1_xpath = (
        "//div[.//p[normalize-space(text())='" + section_title + "'] and "
        " .//p[contains(concat(' ', normalize-space(@class), ' '), ' name ')]]"
    )
    try:
        locator = (By.XPATH, fb1_xpath)
        header = safe_find(driver, locator, timeout=timeout)
        WebDriverWait(driver, timeout).until(EC.visibility_of(header))
        WebDriverWait(driver, timeout).until(lambda d: header.is_displayed() and header.is_enabled())
        logging.info("Section header located via fallback_1 XPath: %s", fb1_xpath)
        return header, fb1_xpath, "fallback_1"
    except (TimeoutException, NoSuchElementException) as e:
        logging.info("Fallback_1 header locator failed for '%s': %s", section_title, e)

    # Attempt 3: Any element with exact text, then nearest ancestor div as header container
    anchor_xpath = f"//*[self::p or self::div or self::span or self::a][normalize-space(text())='{section_title}']"
    try:
        anchor_el = safe_find(driver, (By.XPATH, anchor_xpath), timeout=timeout)
        WebDriverWait(driver, timeout).until(EC.visibility_of(anchor_el))
        # Prefer the nearest ancestor div as the clickable header region
        fb2_xpath = f"{anchor_xpath}/ancestor::div[1]"
        try:
            header = safe_find(driver, (By.XPATH, fb2_xpath), timeout=timeout)
            WebDriverWait(driver, timeout).until(lambda d: header.is_displayed() and header.is_enabled())
            logging.info("Section header located via fallback_2 ancestor div: %s", fb2_xpath)
            return header, fb2_xpath, "fallback_2"
        except (TimeoutException, NoSuchElementException):
            # Use the anchor element itself as a last resort
            header = anchor_el
            logging.info("Section header approximated via fallback_2 anchor element: %s", anchor_xpath)
            return header, anchor_xpath, "fallback_2_anchor"
    except (TimeoutException, NoSuchElementException) as e:
        logging.error("All header locator strategies failed for '%s': %s", section_title, e)
        raise
