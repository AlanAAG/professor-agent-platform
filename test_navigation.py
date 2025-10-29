"""
Standalone E2E navigation test for Selenium selectors and logic.
Flow: Login -> Navigate to course -> Click Resources -> Expand section -> Validate items list.

Run directly: python test_navigation.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)

# Ensure project root is on sys.path so `src.*` imports work when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project modules
from src.harvester import navigation, config  # noqa: E402  (import after sys.path mutation)


# --- Configuration and Setup ---
def setup_test_environment() -> None:
    """Configure logging, load environment variables, and prepare directories."""
    # Load .env early for credentials and settings
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    # Ensure screenshot directory exists for any upstream failures
    try:
        Path(config.SETTINGS.screenshot_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        # Non-fatal if this fails; we'll still try saving screenshots to CWD when needed
        pass

    # Console logger (INFO level) for high-level test progress
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # Root logger setup
    logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)

    # Selenium logger (DEBUG) to file for detailed driver output
    selenium_log_path = PROJECT_ROOT / "selenium_test.log"
    selenium_file_handler = logging.FileHandler(selenium_log_path, mode="w", encoding="utf-8")
    selenium_file_handler.setLevel(logging.DEBUG)
    selenium_file_handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    selenium_logger = logging.getLogger("selenium")
    selenium_logger.setLevel(logging.DEBUG)
    selenium_logger.handlers = [selenium_file_handler]
    selenium_logger.propagate = False  # Avoid duplicating into console output

    logging.info("Test environment configured. Selenium debug -> %s", selenium_log_path)


# --- Main Test Flow ---
def run_navigation_test() -> None:
    """Run the end-to-end navigation test for course PRTC301."""
    # Constants for the test
    TEST_COURSE_CODE = "PRTC301"
    TEST_SECTION_TITLE = "Session Recordings"
    TEST_GROUP_NAME: Optional[str] = None

    # Use configured group name if available in mapping
    try:
        TEST_GROUP_NAME = config.COURSE_MAP.get(TEST_COURSE_CODE, {}).get("group")
    except Exception:
        TEST_GROUP_NAME = None

    # Derived selectors for better error messages
    course_link_xpath = config.COURSE_LINK_XPATH_TEMPLATE.format(course_code=TEST_COURSE_CODE)
    group_header_xpath = (
        config.GROUP_HEADER_XPATH_TEMPLATE.format(group_name=TEST_GROUP_NAME)
        if TEST_GROUP_NAME
        else None
    )
    resources_tab_xpath = config.RESOURCES_TAB_XPATH
    section_header_xpath = config.SECTION_HEADER_XPATH_TPL.format(section_title=TEST_SECTION_TITLE)
    items_xpath = f"{section_header_xpath}/following-sibling::div[1]//{config.RESOURCE_ITEM_CSS}"

    overall_success = False
    items_count = 0

    logging.info("STEP 1: Initializing driver and logging in ...")
    print("[1/5] Login: starting...")

    try:
        with navigation.launch_and_login() as driver:
            print("[1/5] Login: PASSED")

            # STEP 2: Navigate to course
            logging.info("STEP 2: Navigating to course %s (group=%s)", TEST_COURSE_CODE, TEST_GROUP_NAME)
            print(f"[2/5] Navigate: opening course {TEST_COURSE_CODE}...")
            try:
                navigation.find_and_click_course_link(driver, TEST_COURSE_CODE, TEST_GROUP_NAME)
                print("[2/5] Navigate: PASSED")
            except TimeoutException as e:
                logging.error(
                    "Timeout locating course link. course_xpath=%s group_xpath=%s error=%s",
                    course_link_xpath,
                    group_header_xpath,
                    e,
                )
                raise
            except NoSuchElementException as e:
                logging.error(
                    "Course link not found. course_xpath=%s group_xpath=%s error=%s",
                    course_link_xpath,
                    group_header_xpath,
                    e,
                )
                raise
            except (WebDriverException, RuntimeError) as e:
                logging.error("Driver error during course navigation: %s", e)
                raise

            # STEP 3: Click Resources tab
            logging.info("STEP 3: Clicking 'Resources' tab ...")
            print("[3/5] Click Tab: opening 'Resources'...")
            try:
                success = navigation.navigate_to_resources_section(driver)
                if not success:
                    # Normalize into a timeout for consistent handling
                    raise TimeoutException(f"Resources tab not clickable using XPATH: {resources_tab_xpath}")
                print("[3/5] Click Tab: PASSED")
            except TimeoutException as e:
                logging.error("Timeout clicking Resources tab. xpath=%s error=%s", resources_tab_xpath, e)
                raise
            except NoSuchElementException as e:
                logging.error("Resources tab element not found. xpath=%s error=%s", resources_tab_xpath, e)
                raise
            except (WebDriverException, RuntimeError) as e:
                logging.error("Driver error on Resources tab click: %s", e)
                raise

            # STEP 4: Expand section and get items
            logging.info("STEP 4: Expanding section '%s' and collecting items ...", TEST_SECTION_TITLE)
            print(f"[4/5] Expand Section: '{TEST_SECTION_TITLE}'...")
            try:
                container_xpath, items = navigation.expand_section_and_get_items(
                    driver, TEST_SECTION_TITLE
                )
                # Validation per requirement: ensure items list is not None and log the count
                assert items is not None, "expand_section_and_get_items() returned None for items"
                items_count = len(items)
                print(f"[4/5] Expand Section: PASSED (items found: {items_count})")
            except TimeoutException as e:
                logging.error(
                    "Timeout expanding section or loading items. header_xpath=%s items_xpath=%s error=%s",
                    section_header_xpath,
                    items_xpath,
                    e,
                )
                raise
            except NoSuchElementException as e:
                logging.error(
                    "Section header or items not found. header_xpath=%s items_xpath=%s error=%s",
                    section_header_xpath,
                    items_xpath,
                    e,
                )
                raise
            except (WebDriverException, RuntimeError) as e:
                logging.error("Driver error expanding section: %s", e)
                # Screenshot for generic failure at final step (crucial for headless debugging)
                screenshot_path = PROJECT_ROOT / f"nav_failure_{TEST_COURSE_CODE}.png"
                try:
                    driver.save_screenshot(str(screenshot_path))
                    print(f"[!] Failure screenshot saved to: {screenshot_path}")
                except Exception as se:
                    logging.error("Failed to save failure screenshot: %s", se)
                raise
            except Exception as e:
                # Generic fallback for unexpected issues at final step (also save screenshot)
                logging.error("Unexpected error during section expansion: %s", e)
                screenshot_path = PROJECT_ROOT / f"nav_failure_{TEST_COURSE_CODE}.png"
                try:
                    driver.save_screenshot(str(screenshot_path))
                    print(f"[!] Failure screenshot saved to: {screenshot_path}")
                except Exception as se:
                    logging.error("Failed to save failure screenshot: %s", se)
                raise

            # STEP 5: Final validation and status
            logging.info("STEP 5: Validation and result reporting ...")
            print("[5/5] Items Found: validating non-null list...")
            try:
                # Already asserted non-None; log count again for clarity
                logging.info("Retrieved %d items from '%s' section.", items_count, TEST_SECTION_TITLE)
                overall_success = True
                print(f"SUCCESS: Retrieved {items_count} items from '{TEST_SECTION_TITLE}'.")
            except Exception as e:
                logging.error("Validation failed: %s", e)
                raise

    except TimeoutException as e:
        print(f"FAILURE: Timeout occurred. Details logged. Error: {e}")
    except NoSuchElementException as e:
        print(f"FAILURE: Element not found. Details logged. Error: {e}")
    except (WebDriverException, RuntimeError) as e:
        print(f"FAILURE: Critical WebDriver or runtime error. Details logged. Error: {e}")
    except Exception as e:
        print(f"FAILURE: Unexpected error. Details logged. Error: {e}")

    # Final console status (in addition to prints above)
    if overall_success:
        logging.info("Navigation test completed successfully. Item count=%d", items_count)
    else:
        logging.info("Navigation test failed. See logs and selenium_test.log for details.")


if __name__ == "__main__":
    setup_test_environment()
    run_navigation_test()
