# src/harvester/navigation.py

import logging
import os
import random
import time
from typing import Optional, Iterator
from urllib.parse import urlparse, parse_qs
from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    ElementClickInterceptedException,
)

from . import config
# --- Screenshot helper ---
def _save_screenshot(driver: webdriver.Chrome, filename: str) -> None:
    try:
        dir_path = getattr(config.SETTINGS, "screenshot_dir", "logs/error_screenshots")
        os.makedirs(dir_path, exist_ok=True)
        full_path = os.path.join(dir_path, filename)
        driver.save_screenshot(full_path)
    except Exception:
        pass



# --- Driver setup ---
def _create_driver() -> webdriver.Chrome:
    """
    MODIFIED: Simplified this function to match the working Colab script.
    Removed all anti-bot detection and experimental options.
    """
    options = webdriver.ChromeOptions()
    
    # Check headless setting
    try:
        headless = bool(getattr(config.SETTINGS, "selenium_headless", True))
    except Exception:
        env_val = (os.environ.get("HARVESTER_SELENIUM_HEADLESS") or os.environ.get("SELENIUM_HEADLESS") or "true").strip().lower()
        headless = env_val in ("1", "true", "yes")
    
    # Use the same arguments as the working Colab script
    if headless:
        # Use "--headless" (older) or "--headless=new" (newer).
        # The Colab script used "--headless", let's stick to that for compatibility.
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Add back a few useful (but non-suspicious) options
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1440,900")
    
    # Allow CI/providers to specify Chrome binary
    chrome_binary = (
        os.environ.get("CHROME_PATH")
        or os.environ.get("GOOGLE_CHROME_SHIM")
        or os.environ.get("CHROME_BIN")
    )
    if chrome_binary:
        options.binary_location = chrome_binary

    driver = webdriver.Chrome(options=options)

    # Set timeouts
    try:
        driver.implicitly_wait(0) # We use explicit waits
    except Exception:
        pass
    try:
        driver.set_page_load_timeout(int(getattr(config.SETTINGS, "page_load_timeout", 60)))
    except Exception:
        driver.set_page_load_timeout(60)

    # Log versions
    try:
        caps = getattr(driver, "capabilities", {}) or {}
        browser_version = caps.get("browserVersion") or caps.get("version")
        chrome_info = caps.get("chrome") or {}
        chromedriver_version = (chrome_info.get("chromedriverVersion") or "").split(" ")[0]
        logging.info(f"Selenium session (Simple Mode): Chrome {browser_version} / chromedriver {chromedriver_version}")
    except Exception:
        pass
    return driver


def _wait(driver: webdriver.Chrome, seconds: int = 60) -> WebDriverWait:
    """Increased default wait to 60 seconds."""
    return WebDriverWait(driver, seconds)


# --- Cookie/session restore ---
def _try_restore_cookies(driver: webdriver.Chrome) -> bool:
    """Best-effort: load cookies from AUTH_STATE_FILE to bypass login.

    Returns True if cookies were loaded and applied; False otherwise.
    """
    try:
        state_path = getattr(config, "AUTH_STATE_FILE", None) or ""
        if not state_path or not os.path.exists(state_path):
            return False
        import json
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        cookies = data.get("cookies") or []
        if not cookies:
            return False
        # Must be on the target domain before adding cookies
        driver.get(config.BASE_URL)
        # Normalize and add cookies
        applied = 0
        for ck in cookies:
            try:
                cookie = dict(ck)
                # Selenium expects 'expiry' as int seconds since epoch
                if "expires" in cookie and isinstance(cookie["expires"], (int, float)):
                    cookie["expiry"] = int(cookie.pop("expires"))
                # Some fields may cause set-cookie to fail; attempt graceful fallback
                for attempt in range(2):
                    try:
                        driver.add_cookie(cookie)
                        applied += 1
                        break
                    except Exception:
                        # Remove less critical attributes and retry once
                        for k in ["sameSite", "httpOnly", "secure"]:
                            cookie.pop(k, None)
                
            except Exception:
                continue
        # Navigate to courses to validate session
        driver.get(config.COURSES_URL)
        try:
            _wait(driver, 15).until(
                EC.any_of(
                    EC.url_contains("/courses"),
                    EC.visibility_of_element_located((By.XPATH, config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=next(iter(config.DEFAULT_VISIBLE_COURSES)))))
                )
            )
        except Exception:
            pass
        return applied > 0
    except Exception:
        return False


# --- Resilient element helpers (handle stale elements) ---
def _scroll_into_view_center(driver: webdriver.Chrome, element) -> None:
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
    except Exception:
        # Non-fatal; scrolling is best-effort
        pass


def safe_find(
    driver: webdriver.Chrome,
    locator: tuple[str, str],
    timeout: int = 60,
    attempts: int = 3,
    clickable: bool = False,
):
    """Increased default timeout to 60 seconds."""
    last_err: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            condition = EC.element_to_be_clickable(locator) if clickable else EC.presence_of_element_located(locator)
            element = _wait(driver, timeout).until(condition)
            # Touch a property to assert non-stale
            _ = element.is_enabled()
            return element
        except StaleElementReferenceException as e:
            last_err = e
            time.sleep(0.5) # Brief pause before retry on stale
            continue
        except TimeoutException as e:
            last_err = e
            break
    if last_err:
        raise last_err
    raise NoSuchElementException(f"Element not found for locator: {locator}")


def safe_find_all(
    driver: webdriver.Chrome,
    locator: tuple[str, str],
    timeout: int = 60,
    attempts: int = 2,
):
    """Increased default timeout to 60 seconds."""
    last_err: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            _wait(driver, timeout).until(EC.presence_of_element_located(locator))
            elements = driver.find_elements(*locator)
            # Touch each element lightly to ensure not stale
            for el in elements:
                try:
                    _ = el.tag_name
                except StaleElementReferenceException:
                    raise
            return elements
        except StaleElementReferenceException as e:
            last_err = e
            continue
        except TimeoutException as e:
            last_err = e
            break
    if last_err:
        raise last_err
    return []


def safe_click(
    driver: webdriver.Chrome,
    locator: tuple[str, str],
    attempts: int = 3,
    timeout: int = 60,
    scroll: bool = True,
) -> None:
    """
    Robust click: scrolls, pauses, and uses a JS click.
    """
    last_err: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            # 1. Wait for the element to be at least present/clickable
            el = safe_find(driver, locator, timeout=timeout, attempts=1, clickable=True)
            
            # 2. Scroll into view
            if scroll:
                _scroll_into_view_center(driver, el)
                
            # 3. Pause (as seen in working Colab script)
            time.sleep(1) 
            
            # 4. Re-find element just in case scroll made it stale (quick)
            el = safe_find(driver, locator, timeout=min(timeout, 10), attempts=1, clickable=False) 
            
            # 5. Click with JavaScript (more robust)
            driver.execute_script("arguments[0].click();", el)
            return # Success
        except (StaleElementReferenceException, ElementClickInterceptedException) as e:
            last_err = e
            time.sleep(1) # Wait a bit before retrying
            continue
        except TimeoutException as e:
            last_err = e
            break
    if last_err:
        raise last_err


# --- Session and Login ---
def is_session_valid(driver: webdriver.Chrome) -> bool:
    logging.info("Checking if session is valid by navigating to Courses page...")
    try:
        driver.get(config.COURSES_URL)
        # Wait for navigation result (either courses page or redirected to login)
        try:
            _wait(driver, 20).until(
                EC.any_of(
                    EC.url_contains("/courses"),
                    EC.url_contains("/login"),
                )
            )
        except Exception:
            pass  # Fallback to checks below
        if "/login" in (driver.current_url or ""):
            logging.warning("Session invalid: redirected to /login")
            return False

        # Verify visible content on the courses page
        logging.info("Verifying session by looking for dashboard or course content...")
        
        first_default = next(iter(config.DEFAULT_VISIBLE_COURSES))
        link_xpath = config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=first_default)

        # Increased timeout from 20 to 60
        _wait(driver, 60).until(
            EC.any_of(
                EC.visibility_of_element_located((By.XPATH, link_xpath)),
                EC.visibility_of_element_located((By.CSS_SELECTOR, config.DASHBOARD_INDICATOR_CSS))
            )
        )
        logging.info("Session valid: dashboard or course content is visible.")
        return True
    except Exception as e:
        logging.warning(f"Session validation failed: {e}")
        try:
            os.makedirs("logs/error_screenshots", exist_ok=True)
            _save_screenshot(driver, "session_validation_fail.png")
        except Exception:
            pass
        return False


def perform_login(driver: webdriver.Chrome) -> bool:
    os.makedirs("logs/error_screenshots", exist_ok=True)
    username = os.environ.get("COACH_USERNAME")
    password = os.environ.get("COACH_PASSWORD")
    if not username or not password:
        raise ValueError("COACH_USERNAME or COACH_PASSWORD not set in environment")

    logging.info(f"Navigating to login page: {config.LOGIN_URL}")
    driver.get(config.LOGIN_URL)
    # Rely on form field visibility rather than fixed sleep

    try:
        _wait(driver, 25).until(EC.visibility_of_element_located((By.CSS_SELECTOR, config.USERNAME_BY[1])))
        driver.find_element(By.CSS_SELECTOR, config.USERNAME_BY[1]).send_keys(username)
        driver.find_element(By.CSS_SELECTOR, config.PASSWORD_BY[1]).send_keys(password)
        driver.find_element(By.CSS_SELECTOR, config.LOGIN_BUTTON_BY[1]).click()

        # Wait for redirect away from login (allow / or /courses)
        try:
            _wait(driver, 60).until(
                EC.any_of(
                    EC.url_contains("/courses"),
                    EC.url_to_be(config.BASE_URL),
                )
            )
        except TimeoutException:
            # Fallbacks: dashboard indicator or any visible course link
            try:
                _wait(driver, 20).until(
                    EC.any_of(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, config.DASHBOARD_INDICATOR_CSS)),
                        EC.visibility_of_element_located((By.XPATH, config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=next(iter(config.DEFAULT_VISIBLE_COURSES)))))
                    )
                )
            except TimeoutException:
                if "/login" in (driver.current_url or ""):
                    raise RuntimeError("Still on /login after submit")

        if "/login" in (driver.current_url or ""):
            raise RuntimeError("Login appears to have failed")

        logging.info("Login successful")
        # Optional: save cookies
        try:
            cookies = driver.get_cookies()
            os.makedirs(os.path.dirname(config.AUTH_STATE_FILE) or ".", exist_ok=True)
            import json
            with open(config.AUTH_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({"cookies": cookies}, f)
        except Exception:
            pass
        return True
    except Exception as e:
        logging.error(f"Login failed: {e}")
        try:
            _save_screenshot(driver, "login_error.png")
        except Exception:
            pass
        raise


@contextmanager
def launch_and_login() -> Iterator[webdriver.Chrome]:
    """Context-managed Selenium driver that ensures an authenticated session.

    Usage:
        with launch_and_login() as driver:
            ...
    """
    driver = _create_driver()
    try:
        # Attempt cookie-based restore before interactive login
        _try_restore_cookies(driver)
        if not is_session_valid(driver):
            if not (perform_login(driver) and is_session_valid(driver)):
                raise RuntimeError("Failed to establish an authenticated session")
        yield driver
    finally:
        try:
            driver.quit()
        except Exception:
            pass


# --- Course Navigation ---
def find_and_click_course_link(driver: webdriver.Chrome, course_code: str, group_name: Optional[str]) -> None:
    logging.info(f"Navigating to course: {course_code} (group: {group_name or 'unknown'})")
    driver.get(config.COURSES_URL)
    # Wait for either courses or a redirect to login
    try:
        _wait(driver, 20).until(
            EC.any_of(
                EC.url_contains("/courses"),
                EC.url_contains("/login"),
            )
        )
    except Exception:
        pass
    if "/login" in (driver.current_url or ""):
        raise RuntimeError("Not authenticated: redirected to login when opening courses page")

    # Basic visibility check
    try:
        first_default = next(iter(config.DEFAULT_VISIBLE_COURSES))
        check_xpath = config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=first_default)
        # Increased timeout from 10 to 60
        _wait(driver, 60).until(EC.visibility_of_element_located((By.XPATH, check_xpath)))
    except Exception:
        logging.warning("Courses content not immediately visible; waiting briefly for elements")
        try:
            # Increased timeout from 5 to 10
            _wait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, check_xpath)))
        except Exception:
            logging.error("Could not find any visible courses. Page may be empty or changed.")
            pass # Don't raise here, let the next block try

    # Strategy: try direct link first; if not clickable, expand the group and retry
    target_xpath = config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=course_code)
    try:
        logging.info("Trying direct course link click...")
        safe_click(driver, (By.XPATH, target_xpath))
        _wait(driver, 60).until(EC.url_contains("/course"))
        logging.info(f"Opened course {course_code}")
        return
    except Exception as direct_err:
        logging.info(f"Direct click failed for {course_code} ({direct_err}). Attempting via group expansion...")

    try:
        # Expand group then click
        effective_group = group_name or (config.COURSE_MAP.get(course_code, {}).get("group"))
        if not effective_group:
            logging.warning(f"No group defined for {course_code}; retrying direct link click as fallback")
            safe_click(driver, (By.XPATH, target_xpath))
            _wait(driver, 60).until(EC.url_contains("/course"))
            logging.info(f"Opened course {course_code}")
            return

        header_xpath = config.GROUP_HEADER_XPATH_TEMPLATE.format(group_name=effective_group)
        safe_click(driver, (By.XPATH, header_xpath))
        logging.info(f"Expanded group '{effective_group}'")
        logging.info("Pausing for 5 seconds to allow course group to expand...")
        time.sleep(5)

        # Retry clicking course link after expansion
        try:
            safe_click(driver, (By.XPATH, target_xpath))
            _wait(driver, 60).until(EC.url_contains("/course"))
            logging.info(f"Opened course {course_code}")
            return
        except Exception as click_after_expand_err:
            logging.info(f"Click after expanding group failed for {course_code} ({click_after_expand_err}). Trying href navigation fallback...")

        # Fallback: navigate directly via the link's href
        try:
            link_el = safe_find(driver, (By.XPATH, target_xpath), timeout=15, attempts=2)
            href = link_el.get_attribute("href") if link_el else None
            if href:
                if not href.startswith(("http://", "https://")):
                    href = config.BASE_URL.rstrip('/') + '/' + href.lstrip('/')
                logging.info(f"Opening course via href: {href}")
                driver.get(href)
                # Wait for course details page heuristics
                _wait(driver, 60).until(
                    EC.any_of(
                        EC.url_contains("courseCode="),
                        EC.visibility_of_element_located((By.XPATH, config.RESOURCES_TAB_XPATH)),
                    )
                )
                logging.info(f"Opened course {course_code}")
                return
        except Exception as href_nav_err:
            logging.info(f"Href navigation fallback failed for {course_code} ({href_nav_err}).")
    except Exception as e:
        logging.error(f"Course navigation failed for {course_code}: {type(e).__name__}: {e}")
        try:
            _save_screenshot(driver, f"course_nav_error_{course_code}.png")
        except Exception:
            pass
        raise


# --- Resource Navigation (within a course page) ---
def navigate_to_resources_section(driver: webdriver.Chrome) -> bool:
    logging.info("Navigating to Resources section...")
    try:
        safe_click(driver, (By.XPATH, config.RESOURCES_TAB_XPATH))

        # Wait for any section header
        any_selector = (
            config.SECTION_HEADER_XPATH_TPL.format(section_title=config.PRE_READ_SECTION_TITLE)
            + " | "
            + config.SECTION_HEADER_XPATH_TPL.format(section_title=config.IN_CLASS_SECTION_TITLE)
            + " | "
            + config.SECTION_HEADER_XPATH_TPL.format(section_title=config.POST_CLASS_SECTION_TITLE)
            + " | "
            + config.SECTION_HEADER_XPATH_TPL.format(section_title=config.SESSION_RECORDINGS_SECTION_TITLE)
        )
        # Increased timeout from 20 to 60
        _wait(driver, 60).until(EC.visibility_of_element_located((By.XPATH, any_selector)))
        logging.info("Resources section loaded")
        return True
    except Exception as e:
        logging.error(f"Failed to open Resources section: {e}")
        try:
            _save_screenshot(driver, "resources_nav_error.png")
        except Exception:
            pass
        return False


def expand_section_and_get_items(driver: webdriver.Chrome, section_title: str):
    """Click a Resources accordion section and return (container_xpath, item elements).

    Returning the container XPath allows callers to re-fetch items by index to avoid
    StaleElementReferenceException when iterating dynamically-loaded content.
    """
    header_xpath = config.SECTION_HEADER_XPATH_TPL.format(section_title=section_title)
    safe_click(driver, (By.XPATH, header_xpath))
    
    # Add a small pause after clicking the section header
    time.sleep(2) 
    
    # Items are in next sibling div. Wait for it to populate.
    container_xpath = header_xpath + "/parent::div/following-sibling::div[1]"
    _wait(driver, 10).until(EC.presence_of_element_located((By.XPATH, container_xpath)))
    container = driver.find_element(By.XPATH, container_xpath)
    # Wait until at least one resource item is present (best-effort)
    try:
        _wait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, config.RESOURCE_ITEM_CSS))
        )
    except TimeoutException:
        pass
    items = container.find_elements(By.CSS_SELECTOR, config.RESOURCE_ITEM_CSS)
    return container_xpath, items


def get_available_course_codes(driver: webdriver.Chrome) -> set[str]:
    """Detect available course codes on the Courses page by scanning anchor hrefs.

    Expands visible groups to surface lazy-loaded links. Returns a set of codes.
    """
    logging.info("Detecting available courses on the Courses page...")
    codes: set[str] = set()

    driver.get(config.COURSES_URL)
    try:
        _wait(driver, 20).until(
            EC.any_of(
                EC.url_contains("/courses"),
                EC.url_contains("/login"),
            )
        )
    except Exception:
        pass
    if "/login" in (driver.current_url or ""):
        logging.warning("Cannot detect courses: redirected to login")
        return codes

    def _collect_codes_from_links() -> None:
        try:
            links = safe_find_all(driver, (By.XPATH, "//a[contains(@href, 'courseCode=')]"), timeout=10, attempts=2)
        except Exception:
            links = []
        for a in links:
            try:
                href = a.get_attribute("href") or ""
                if not href:
                    continue
                parsed = urlparse(href)
                q = parse_qs(parsed.query or "")
                code = (q.get("courseCode") or [None])[0]
                if not code:
                    # Fallback: manual parse
                    marker = "courseCode="
                    if marker in href:
                        frag = href.split(marker, 1)[1]
                        code = frag.split("&", 1)[0]
                if code:
                    codes.add(code)
            except Exception:
                continue

    # Initial collection from default-visible area
    _collect_codes_from_links()

    # Expand any visible groups to load more courses, then collect again
    try:
        group_titles = [
            el.text.strip()
            for el in driver.find_elements(By.XPATH, "//div[contains(@class,'domainHeader')]//p[contains(@class,'title')]")
            if (el.text or "").strip()
        ]
        for group_name in group_titles:
            try:
                header_xpath = config.GROUP_HEADER_XPATH_TEMPLATE.format(group_name=group_name)
                safe_click(driver, (By.XPATH, header_xpath))
                time.sleep(1)
                _collect_codes_from_links()
            except Exception:
                continue
    except Exception:
        pass

    logging.info(f"Detected {len(codes)} available courses on page: {sorted(codes)}")
    return codes
    