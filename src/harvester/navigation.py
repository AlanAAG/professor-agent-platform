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
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple, Optional
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
)
from . import config

# --- Globals and Initialization ---

# Cache to track which courses have been processed in the current pipeline run.
# Used to prevent navigating to a course link twice if it appears in multiple groups.
_COURSE_LINKS_SEEN_CACHE: set[str] = set()

def reset_course_tracking():
    """Resets the seen courses cache for a fresh pipeline run."""
    _COURSE_LINKS_SEEN_CACHE.clear()


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


def _get_locator_by(locator_tuple: Tuple[str, str]) -> Tuple[By, str]:
    """Transform config-style locator tuples into explicit Selenium By tuples.

    Supports simple methods like "css" and "xpath"; defaults to CSS selector.
    """
    method, value = locator_tuple
    if method == "css":
        return By.CSS_SELECTOR, value
    if method == "xpath":
        return By.XPATH, value
    # Default fallback to CSS
    return By.CSS_SELECTOR, value


def perform_login(driver: webdriver.Chrome) -> bool:
    """Performs the login sequence using stored credentials, with retry logic."""
    username = os.getenv("COACH_USERNAME")
    password = os.getenv("COACH_PASSWORD")
    if not username or not password:
        logging.critical("COACH_USERNAME or COACH_PASSWORD environment variable not set.")
        return False

    MAX_LOGIN_ATTEMPTS = 3

    # Define locators outside the loop for efficiency
    username_locator = _get_locator_by(config.USERNAME_BY)
    password_locator = _get_locator_by(config.PASSWORD_BY)
    login_button_locator = _get_locator_by(config.LOGIN_BUTTON_BY)

    for attempt in range(1, MAX_LOGIN_ATTEMPTS + 1):
        try:
            logging.info(f"Attempting fresh login (Attempt {attempt}/{MAX_LOGIN_ATTEMPTS})...")

            # 1. Navigate (or re-navigate) to Login Page
            driver.get(config.LOGIN_URL)

            # 2. Wait for the login form elements to be present
            WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
                EC.presence_of_element_located(username_locator)
            )

            # 3. Enter credentials
            driver.find_element(*username_locator).send_keys(username)
            driver.find_element(*password_locator).send_keys(password)

            # 4. Click login button
            safe_click(driver, login_button_locator, timeout=config.SETTINGS.wait_timeout)

            # --- Use an extended wait for post-login redirection ---
            EXTENDED_WAIT_TIMEOUT = 45

            # 5. Wait for redirection to the dashboard (or timeout)
            WebDriverWait(driver, EXTENDED_WAIT_TIMEOUT).until(
                EC.url_contains(config.COURSES_URL)
            )

            logging.info(f"Login successful after {attempt} attempt(s).")
            _save_session_state(driver)
            return True

        except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
            logging.warning(f"Login attempt {attempt} failed (Element/Timeout): {e}. Retrying.")
            time.sleep(2)
            continue

        except Exception as e:
            if "Timed out receiving message from renderer" in str(e):
                logging.warning(
                    f"Login attempt {attempt} failed (Renderer Timeout). Pausing and retrying."
                )
                time.sleep(5)
                continue
            else:
                logging.error(f"Login failed on attempt {attempt} due to critical error: {e}")
                break

    logging.error(f"Login failed after {MAX_LOGIN_ATTEMPTS} attempts.")
    driver.save_screenshot(os.path.join(config.SETTINGS.screenshot_dir, "login_failure.png"))
    return False


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
        # Guarantee driver cleanup
        driver.quit()
        logging.info("WebDriver quit.")


# --- Core Navigation Helpers ---

def safe_find(
    driver: webdriver.Chrome,
    locator: Tuple[str, str],
    timeout: int = 30,
    clickable: bool = False,
) -> WebElement:
    """Robust element finding with explicit waits."""
    wait = WebDriverWait(driver, timeout)
    condition = EC.element_to_be_clickable(locator) if clickable else EC.presence_of_element_located(locator)

    # Simple retry logic for StaleElementReferenceException
    for attempt in range(3):
        try:
            return wait.until(condition)
        except StaleElementReferenceException:
            if attempt < 2:
                time.sleep(1)
                continue
            raise
        except (TimeoutException, NoSuchElementException) as e:
            # --- START CRITICAL ERROR LOGGING ---
            locator_string = f"{locator[0]}_{locator[1].replace('//', '').replace('/', '_').replace('[', '_').replace(']', '')[:50]}"
            _take_error_screenshot(driver, f"find_timeout_{locator_string}")
            # --- END CRITICAL ERROR LOGGING ---
            raise TimeoutException(f"Timed out waiting for element located by {locator}. Original Error: {e}")
            
    # Keep the final implicit raise, though the new except block above covers most failures
    raise TimeoutException(f"Timed out waiting for element located by {locator}") 


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
    # Small pause to allow UI transition
    time.sleep(1.5)


def get_available_course_codes(driver: webdriver.Chrome) -> set[str]:
    """Scrapes the course dashboard for all course codes visible to the user."""
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


def find_and_click_course_link(driver: webdriver.Chrome, course_code: str, group_name: str | None = None):
    """Navigate from the courses dashboard into a specific course page.

    Logic mirrors the stable Colab flow:
      - If the course is among default-visible cards, click it directly.
      - Otherwise, expand the owning group header first, then click the link.

    Uses JS-based clicks and scroll-into-view to maximize reliability.
    """

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
            course_link = safe_find(driver, course_locator, timeout=config.SETTINGS.wait_timeout)
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", course_link)
            time.sleep(1)
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
                    group_header = safe_find(driver, group_locator, timeout=15)
                    # Scroll and click to expand
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", group_header)
                    time.sleep(1)
                    driver.execute_script("arguments[0].click();", group_header)
                    logging.info("Clicked group header '%s'", resolved_group)
                    # Allow content to render
                    time.sleep(3)
                except TimeoutException:
                    logging.warning(
                        "Group header '%s' not found or not clickable. Proceeding with full-page search.",
                        resolved_group,
                    )

            # Now find and click the course link
            course_link = safe_find(driver, course_locator, timeout=config.SETTINGS.wait_timeout)
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", course_link)
            time.sleep(1)
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


def navigate_to_resources_section(driver: webdriver.Chrome) -> bool:
    """Click the "Resources" tab on the course page, with selector fallbacks.

    Attempts the primary repository selector first, then tries alternate
    partner-provided selectors from the stable Colab script.
    """
    # Primary selector from config
    primary_locator = (By.XPATH, config.RESOURCES_TAB_XPATH)
    # Fallbacks seen in partner Colab scripts/UI variants
    fallback_locators: List[Tuple[By, str]] = [
        (By.XPATH, "//div[contains(@class, 'sc-Rbkqr')]//h4[contains(text(), 'Resources')]")
    ]

    candidates: List[Tuple[By, str]] = [primary_locator] + fallback_locators

    last_error: Optional[Exception] = None
    for locator in candidates:
        try:
            safe_click(driver, locator, timeout=10)
            time.sleep(2)  # allow content to swap in
            logging.info("Navigated to Resources section via locator: %s", locator)
            return True
        except Exception as e:
            last_error = e
            logging.debug("Resources click attempt failed for %s: %s", locator, e)
            continue

    logging.warning("Resources tab not found or failed to click using all locators. Last error: %s", last_error)
    return False


def expand_section_and_get_items(driver: webdriver.Chrome, section_title: str) -> Tuple[str, List[WebElement]]:
    """Expands a specific resource section and returns its primary link elements (<a>)."""
    
    # 1. Find the section header and click to expand
    section_locator = (By.XPATH, config.SECTION_HEADER_XPATH_TPL.format(section_title=section_title))
    
    try:
        section_header = safe_find(driver, section_locator, timeout=10)
        
        # Click to expand the section (the header acts as the toggle)
        driver.execute_script("arguments[0].click();", section_header)
        logging.info(f"Expanded section: {section_title}")
        # Allow UI to react and begin loading dynamic content
        time.sleep(1)

        # 2. Define and locate the container that holds resource items (following the header)
        container_xpath = f"{section_locator[1]}/following-sibling::div[1]"
        container_el = safe_find(driver, (By.XPATH, container_xpath), timeout=10)

        # 3. Wait for at least one primary link (<a href="...">) to appear within the container
        LINK_CSS_SELECTOR = "a[href]"

        def _links_present(_):
            return container_el.find_elements(By.CSS_SELECTOR, LINK_CSS_SELECTOR)

        try:
            # Increased timeout to improve robustness in headless/slow environments
            WebDriverWait(driver, 15).until(lambda d: len(_links_present(d)) > 0)
        except TimeoutException:
            # It's acceptable for some sections to be empty; normalize to empty list
            return container_xpath, []

        # 4. Retrieve all relevant link elements within the container
        items: List[WebElement] = container_el.find_elements(By.CSS_SELECTOR, LINK_CSS_SELECTOR)

        # Return the XPath to the parent container and the list of item elements
        return container_xpath, items

    except TimeoutException:
        logging.info(f"Section '{section_title}' contains no items or failed to load.")
        return "", []
    except Exception as e:
        logging.error(f"Error expanding section '{section_title}': {e}")
        raise

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


def expand_section_and_get_items(driver: webdriver.Chrome, section_title: str) -> Tuple[str, List[WebElement]]:
    """Expand a resource section header and return all link items within its container.

    Fallback strategy:
    1) Section Header Click (3 attempts):
       - Primary: XPath from config with exact text match on header structure.
       - Fallback 1: Any header-like div containing a <p class="name"> with exact title.
       - Fallback 2: Any element with exact text; click its nearest ancestor div.
       For each: scroll into view, wait visible and clickable, use JS click.

    2) Container Location (2 attempts + pattern assist):
       - Primary: following-sibling::div[1] of the clicked header XPath.
       - Fallback: nearest header wrapper's following sibling; if still not found, search
         parent area for resource-like patterns (classes including 'fileBox' or 'resource').

    3) Link Extraction (comprehensive):
       - Collect all descendant anchors: "div.fileBox a[href], div.fileContentCol a[href], a[href]".
       - Filter empty/self-referential hrefs.

    Returns (container_xpath_used, list_of_anchor_elements). If no content, returns (xpath or '', []).
    """

    # 1) Locate header robustly
    header_el, header_xpath_used, strategy = _locate_section_header_robust(
        driver, section_title, timeout=5
    )

    # Scroll into view and click via JS
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", header_el)
        WebDriverWait(driver, 5).until(lambda d: header_el.is_displayed() and header_el.is_enabled())
        driver.execute_script("arguments[0].click();", header_el)
        logging.info("Clicked section header '%s' via strategy: %s", section_title, strategy)
    except StaleElementReferenceException:
        # Re-locate once on stale and retry click
        logging.info("Header element went stale; re-locating for '%s'", section_title)
        header_el, header_xpath_used, strategy = _locate_section_header_robust(driver, section_title, timeout=5)
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", header_el)
        WebDriverWait(driver, 5).until(lambda d: header_el.is_displayed() and header_el.is_enabled())
        driver.execute_script("arguments[0].click();", header_el)
        logging.info("Clicked (retry) section header '%s' via strategy: %s", section_title, strategy)

    # Allow UI to render the content container
    time.sleep(0.75)

    # 2) Locate the container: Primary attempt - following sibling of the header XPath
    container_xpath_primary = f"{header_xpath_used}/following-sibling::div[1]"
    container_el: Optional[WebElement] = None
    container_xpath_used: str = container_xpath_primary

    try:
        container_el = safe_find(driver, (By.XPATH, container_xpath_primary), timeout=5)
        WebDriverWait(driver, 5).until(EC.visibility_of(container_el))
        logging.info("Container located via primary following-sibling: %s", container_xpath_primary)
    except (TimeoutException, NoSuchElementException) as e:
        logging.info("Primary container location failed; trying fallback for '%s': %s", section_title, e)
        # Fallback: go to the nearest wrapper div around header, then its following sibling
        wrapper_xpath = f"{header_xpath_used}/ancestor::div[1]"
        fallback_container_xpath = f"{wrapper_xpath}/following-sibling::div[1]"
        try:
            container_el = safe_find(driver, (By.XPATH, fallback_container_xpath), timeout=5)
            WebDriverWait(driver, 5).until(EC.visibility_of(container_el))
            container_xpath_used = fallback_container_xpath
            logging.info("Container located via fallback sibling of wrapper: %s", fallback_container_xpath)
        except (TimeoutException, NoSuchElementException) as e2:
            # Last-resort within fallback scope: search parent area for resource-like content
            pattern_xpath = (
                f"({wrapper_xpath}//div[contains(@class,'fileBox') or "
                f" contains(translate(@class,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'resource')]/ancestor::div[1])[1]"
            )
            try:
                container_el = safe_find(driver, (By.XPATH, pattern_xpath), timeout=5)
                WebDriverWait(driver, 5).until(EC.visibility_of(container_el))
                container_xpath_used = pattern_xpath
                logging.info("Container located via resource pattern search: %s", pattern_xpath)
            except (TimeoutException, NoSuchElementException) as e3:
                logging.info(
                    "No visible container found for section '%s' after fallbacks. Returning empty list. Last errors: %s | %s | %s",
                    section_title,
                    e,
                    e2,
                    e3,
                )
                return "", []

    # 3) Extract links comprehensively from the container
    try:
        link_selectors = [
            "div.fileBox a[href]",
            "div.fileContentCol a[href]",
            "a[href]",
        ]
        selector_union = ", ".join(link_selectors)
        anchors: List[WebElement] = container_el.find_elements(By.CSS_SELECTOR, selector_union) if container_el else []

        # Filter out empty or self-referential hrefs
        filtered: List[WebElement] = []
        for a in anchors:
            try:
                href = (a.get_attribute("href") or "").strip()
            except StaleElementReferenceException:
                continue
            if not href:
                continue
            lower_href = href.lower()
            if lower_href.startswith("javascript:") or lower_href == "#":
                continue
            filtered.append(a)

        logging.info(
            "Extracted %d link(s) from section '%s' using container '%s'",
            len(filtered),
            section_title,
            container_xpath_used,
        )
        return container_xpath_used, filtered
    except Exception as e:
        logging.warning("Failed extracting links for '%s': %s", section_title, e)
        return container_xpath_used, []