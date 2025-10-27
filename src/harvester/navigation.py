# src/harvester/navigation.py

import logging
import os
import random
import time
from typing import Optional, Iterator
from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from . import config


# --- Driver setup ---
def _create_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    # Prefer structured settings if available
    headless_env = (
        (os.environ.get("HARVESTER_SELENIUM_HEADLESS") or os.environ.get("SELENIUM_HEADLESS") or "true")
        .strip()
        .lower()
    )
    if headless_env in ("1", "true", "yes"):
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1440,900")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=en-US,en")
    options.add_argument("--accept-lang=en-US,en;q=0.9")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(options=options)
    try:
        # Reduce webdriver fingerprints
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});",
            },
        )
    except Exception:
        # Best-effort; continue if CDP not available
        pass
    try:
        from .config import SETTINGS  # Lazy import to avoid cycles
        driver.set_page_load_timeout(int(getattr(SETTINGS, "page_load_timeout", 60)))
    except Exception:
        driver.set_page_load_timeout(60)
    return driver


def _wait(driver: webdriver.Chrome, seconds: int = 30) -> WebDriverWait:
    return WebDriverWait(driver, seconds)


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
        first_default = next(iter(config.DEFAULT_VISIBLE_COURSES))
        link_xpath = config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=first_default)
        _wait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, link_xpath)))
        logging.info("Session valid: courses page content visible")
        return True
    except Exception as e:
        logging.warning(f"Session validation failed: {e}")
        try:
            os.makedirs("logs/error_screenshots", exist_ok=True)
            driver.save_screenshot("logs/error_screenshots/session_validation_fail.png")
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

        # Wait for redirect away from login
        try:
            _wait(driver, 30).until(EC.url_contains("/courses"))
        except TimeoutException:
            # Fallback: dashboard indicator visible
            try:
                _wait(driver, 15).until(EC.visibility_of_element_located((By.CSS_SELECTOR, config.DASHBOARD_INDICATOR_CSS)))
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
            driver.save_screenshot("logs/error_screenshots/login_error.png")
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
        _wait(driver, 10).until(EC.visibility_of_element_located((By.XPATH, check_xpath)))
    except Exception:
        logging.warning("Courses content not immediately visible; waiting briefly for elements")
        try:
            _wait(driver, 5).until(EC.visibility_of_element_located((By.XPATH, check_xpath)))
        except Exception:
            pass

    try:
        if course_code in config.DEFAULT_VISIBLE_COURSES:
            target_xpath = config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=course_code)
            link_el = _wait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, target_xpath)))
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link_el)
            # Re-validate clickable after scroll (avoids animation issues)
            link_el = _wait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, target_xpath)))
            link_el.click()
            _wait(driver, 30).until(EC.url_contains("/course"))
            logging.info(f"Opened course {course_code}")
            return

        # Expand group then click
        effective_group = group_name or (config.COURSE_MAP.get(course_code, {}).get("group"))
        if not effective_group:
            logging.warning(f"No group defined for {course_code}; trying direct link click")
            target_xpath = config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=course_code)
            link_el = _wait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, target_xpath)))
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link_el)
            link_el = _wait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, target_xpath)))
            link_el.click()
            _wait(driver, 30).until(EC.url_contains("/course"))
            return

        header_xpath = config.GROUP_HEADER_XPATH_TEMPLATE.format(group_name=effective_group)
        header_el = _wait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, header_xpath)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", header_el)
        header_el = _wait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, header_xpath)))
        header_el.click()
        logging.info(f"Expanded group '{effective_group}'")

        target_xpath = config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=course_code)
        # Wait for course link to become clickable after expansion
        link_el = _wait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, target_xpath)))
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link_el)
        link_el = _wait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, target_xpath)))
        link_el.click()
        _wait(driver, 30).until(EC.url_contains("/course"))
        logging.info(f"Opened course {course_code}")
    except Exception as e:
        logging.error(f"Course navigation failed for {course_code}: {e}")
        try:
            driver.save_screenshot(f"logs/error_screenshots/course_nav_error_{course_code}.png")
        except Exception:
            pass
        raise


# --- Resource Navigation (within a course page) ---
def navigate_to_resources_section(driver: webdriver.Chrome) -> bool:
    logging.info("Navigating to Resources section...")
    try:
        resources_el = _wait(driver, 25).until(
            EC.element_to_be_clickable((By.XPATH, config.RESOURCES_TAB_XPATH))
        )
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", resources_el)
        # Re-validate clickable after scroll
        resources_el = _wait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, config.RESOURCES_TAB_XPATH))
        )
        resources_el.click()

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
        _wait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, any_selector)))
        logging.info("Resources section loaded")
        return True
    except Exception as e:
        logging.error(f"Failed to open Resources section: {e}")
        try:
            driver.save_screenshot("logs/error_screenshots/resources_nav_error.png")
        except Exception:
            pass
        return False


def expand_section_and_get_items(driver: webdriver.Chrome, section_title: str):
    """Click a Resources accordion section and return list of item elements within it."""
    header_xpath = config.SECTION_HEADER_XPATH_TPL.format(section_title=section_title)
    header_el = _wait(driver, 7).until(EC.element_to_be_clickable((By.XPATH, header_xpath)))
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", header_el)
    header_el = _wait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, header_xpath)))
    header_el.click()
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
    return items
