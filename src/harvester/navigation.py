"""
Core Selenium navigation logic for the harvester module.
Handles driver initialization, login, session management, and course navigation.
"""

import os
import json
import logging
import time
import re
import shutil
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, List, Tuple, Optional
from selenium import webdriver
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
    """Performs the login sequence using stored credentials."""
    username = os.getenv("COACH_USERNAME")
    password = os.getenv("COACH_PASSWORD")
    if not username or not password:
        logging.critical("COACH_USERNAME or COACH_PASSWORD environment variable not set.")
        return False

    try:
        driver.get(config.LOGIN_URL)

        # Resolve explicit Selenium By locators from config
        username_locator = _get_locator_by(config.USERNAME_BY)
        password_locator = _get_locator_by(config.PASSWORD_BY)
        login_button_locator = _get_locator_by(config.LOGIN_BUTTON_BY)

        # Wait for the login form elements to be present
        WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
            EC.presence_of_element_located(username_locator)
        )

        # Enter credentials
        driver.find_element(*username_locator).send_keys(username)
        driver.find_element(*password_locator).send_keys(password)

        # Click login button (single reliable call)
        safe_click(driver, login_button_locator, timeout=config.SETTINGS.wait_timeout)

        # Wait for redirection to the dashboard (or timeout)
        WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
            EC.url_contains(config.COURSES_URL)
        )

        logging.info("Login successful.")
        _save_session_state(driver)
        return True

    except Exception as e:
        logging.error(f"Login failed: {e}")
        # Save screenshot on failure
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
) -> webdriver.remote.webelement.WebElement:
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
    raise TimeoutException(f"Timed out waiting for element located by {locator}")


def safe_find_all(
    driver: webdriver.Chrome,
    locator: Tuple[str, str],
    timeout: int = 30,
) -> List[webdriver.remote.webelement.WebElement]:
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
    """Navigates to the main course page by finding the course link."""
    
    # 1. Check Cache (Global scope tracking for a single pipeline run)
    if course_code in _COURSE_LINKS_SEEN_CACHE:
        logging.warning(f"Course {course_code} already processed in this run. Skipping navigation to avoid duplication.")
        return
    
    # 2. Navigate to main course dashboard
    driver.get(config.COURSES_URL)
    
    # 3. Find Group Header (if applicable)
    if group_name:
        group_locator = (By.XPATH, config.GROUP_HEADER_XPATH_TEMPLATE.format(group_name=group_name))
        try:
            group_header = safe_find(driver, group_locator, timeout=10)
            logging.info(f"Found group header for: {group_name}")
            # Ensure the group is visible/expanded (by scrolling to it)
            driver.execute_script("arguments[0].scrollIntoView(true);", group_header)
        except TimeoutException:
            logging.warning(f"Group header not found for '{group_name}'. Searching full page.")
            pass # Continue searching without group context

    # 4. Find Course Link (using the course code)
    course_locator = (By.XPATH, config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=course_code))
    try:
        course_link = safe_find(driver, course_locator, timeout=config.SETTINGS.wait_timeout)
        # Use JavaScript to click the link and navigate to the course page
        driver.execute_script("arguments[0].click();", course_link)
        
        # Track successful navigation
        _COURSE_LINKS_SEEN_CACHE.add(course_code)
        
        # Wait for the URL to change/main course page to load
        WebDriverWait(driver, config.SETTINGS.wait_timeout).until(
            EC.url_contains(f"courseCode={course_code}")
        )
        logging.info(f"Successfully navigated to course: {course_code}")

    except TimeoutException:
        logging.error(f"Course link not found or navigation timed out for {course_code}.")
        raise
    except Exception as e:
        logging.error(f"Error clicking course link for {course_code}: {e}")
        raise


def navigate_to_resources_section(driver: webdriver.Chrome) -> bool:
    """Clicks the 'Resources' tab on the course page."""
    try:
        # Wait for the Resources tab to be present and clickable
        safe_click(driver, (By.XPATH, config.RESOURCES_TAB_XPATH), timeout=10)
        
        # Simple wait for the page content to update after the click
        time.sleep(2)
        
        logging.info("Navigated to Resources section.")
        return True
    except TimeoutException:
        logging.warning("Resources tab not found or failed to click.")
        return False


def expand_section_and_get_items(driver: webdriver.Chrome, section_title: str) -> Tuple[str, List[webdriver.remote.webelement.WebElement]]:
    """Expands a specific resource section and returns its item elements."""
    
    # 1. Find the section header and click to expand
    section_locator = (By.XPATH, config.SECTION_HEADER_XPATH_TPL.format(section_title=section_title))
    
    try:
        section_header = safe_find(driver, section_locator, timeout=10)
        
        # Click to expand the section (the header acts as the toggle)
        driver.execute_script("arguments[0].click();", section_header)
        logging.info(f"Expanded section: {section_title}")

        # 2. Define the XPath for the resource items container *relative* to the header's parent
        # The actual file items (fileBox) appear as siblings/descendants after expansion.
        # We look for the common resource list containers that follow the header.
        
        # XPath for the container holding the items
        container_xpath = f"{section_locator[1]}/following-sibling::div[1]"
        
        # Wait for at least one item to appear within the container
        items_locator = (By.XPATH, f"{container_xpath}//{config.RESOURCE_ITEM_CSS}")
        
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located(items_locator)
        )
        
        # 3. Retrieve all items (elements)
        items = driver.find_elements(By.XPATH, f"{container_xpath}//{config.RESOURCE_ITEM_CSS}")
        
        # Return the XPath to the parent container and the list of item elements
        return container_xpath, items

    except TimeoutException:
        logging.info(f"Section '{section_title}' contains no items or failed to load.")
        return "", []
    except Exception as e:
        logging.error(f"Error expanding section '{section_title}': {e}")
        raise