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
    overall_success = False
    total_items_found = 0

    logging.info("STEP 1: Initializing driver and logging in ...")
    print("[1/5] Login: starting...")

    try:
        with navigation.launch_and_login() as driver:
            print("[1/5] Login: PASSED")

            # STEP 2: Navigate to course
            path_hint = (
                "default-visible"
                if TEST_COURSE_CODE in getattr(config, "DEFAULT_VISIBLE_COURSES", set())
                else f"group='{TEST_GROUP_NAME}'"
            )
            logging.info(
                "STEP 2: Navigating to course %s via %s",
                TEST_COURSE_CODE,
                path_hint,
            )
            print(f"[2/5] Navigate: opening course {TEST_COURSE_CODE} via {path_hint}...")
            try:
                navigation.find_and_click_course_link(driver, TEST_COURSE_CODE, TEST_GROUP_NAME)
                print("[2/5] Navigate: PASSED")
                # Store course URL for robust re-navigation before each section expansion
                course_page_url = driver.current_url
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
            logging.info("STEP 3: Clicking 'Resources' tab (with selector fallbacks) ...")
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

            # --- NEW STEP: Define all sections for full coverage ---
            ALL_SECTIONS = [
                config.PRE_READ_SECTION_TITLE,
                config.IN_CLASS_SECTION_TITLE,
                config.POST_CLASS_SECTION_TITLE,
                config.SESSION_RECORDINGS_SECTION_TITLE,
            ]
            all_sections_passed = True

            # --- STEP 4: Loop through all sections ---
            logging.info("STEP 4: Iterating through all resource sections ...")
            for section_title in ALL_SECTIONS:
                print(f"\n[4/{len(ALL_SECTIONS)}] Testing Section: '{section_title}'")

                # Robustness Check: Re-click Resources tab before each expansion
                driver.get(course_page_url)  # Navigate back to main course page
                navigation.navigate_to_resources_section(driver)  # Re-click Resources tab

                try:
                    _container_descriptor, items, header_el = navigation.expand_section_and_get_items(
                        driver,
                        section_title,
                    )
                    if items:
                        print(f"  ✅ SUCCESS: Found {len(items)} item(s).")
                        total_items_found += len(items)
                    else:
                        print("  ⚠️ SUCCESS: Section is accessible but empty (0 items).")
                except Exception as e:
                    print(
                        f"  ❌ FAILURE: Section failed to load/expand. Error: {type(e).__name__}: {e}"
                    )
                    logging.error("Section '%s' failed with error: %s", section_title, e)
                    all_sections_passed = False
                    try:
                        screenshot_name = f"nav_failure_{TEST_COURSE_CODE}_{section_title.replace(' ', '_')}.png"
                        driver.save_screenshot(screenshot_name)
                        print("  -> Screenshot saved for failure analysis.")
                    except Exception as se:
                        logging.error("Failed to save failure screenshot: %s", se)

            # --- FINAL RESULT REPORTING ---
            if all_sections_passed:
                print("\n🎉 ALL SECTIONS PASSED NAVIGATION TEST.")
                print(f"Total resources found across all sections: {total_items_found}")
                overall_success = True
            else:
                print("\n❌ TEST FAILED. One or more sections were inaccessible.")
                overall_success = False

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
        logging.info("Navigation test completed successfully. Total items=%d", total_items_found)
    else:
        logging.info("Navigation test failed. See logs and selenium_test.log for details.")


if __name__ == "__main__":
    setup_test_environment()
    run_navigation_test()
