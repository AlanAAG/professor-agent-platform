# src/harvester/navigation.py

import logging
import os # Added for AUTH_STATE_FILE path
from playwright.async_api import Page, Locator, Playwright, Browser, BrowserContext
from . import config # Imports selectors and URLs from config.py
# Assuming utils.py is in src/shared/
from src.shared import utils # For utility functions if needed later

# --- Authentication & Initialization ---

async def is_session_valid(page: Page) -> bool:
    """Checks if the saved session can successfully load the dashboard."""
    logging.info("Checking if session is valid...")
    try:
        await page.goto(config.BASE_URL, wait_until="domcontentloaded", timeout=15000)
        # Wait for a unique element that ONLY appears after login
        await page.wait_for_selector(config.DASHBOARD_INDICATOR, state="visible", timeout=10000)
        logging.info("Session is valid.")
        return True
    except Exception as e:
        logging.warning(f"Session validation failed: {e}")
        return False

async def perform_login(page: Page):
    """Handles the initial login process."""
    logging.info(f"Navigating to login page: {config.LOGIN_URL}")
    await page.goto(config.LOGIN_URL, wait_until="load")

    logging.info("Attempting login...")
    username = os.environ.get("COACH_USERNAME")
    password = os.environ.get("COACH_PASSWORD")

    if not username or not password:
        logging.critical("COACH_USERNAME or COACH_PASSWORD not set in environment. Cannot login.")
        raise ValueError("Missing login credentials in environment variables.")

    try:
        await page.locator(config.USERNAME_SELECTOR).fill(username)
        await page.locator(config.PASSWORD_SELECTOR).fill(password)
        await page.locator(config.LOGIN_BUTTON_SELECTOR).click()

        # Wait for navigation to dashboard after login
        await page.wait_for_selector(config.DASHBOARD_INDICATOR, state="visible", timeout=30000) # Increased timeout
        logging.info("Login successful!")
        # Save authentication state
        await page.context.storage_state(path=config.AUTH_STATE_FILE)
        logging.info(f"Authentication state saved to {config.AUTH_STATE_FILE}")
        return True
    except Exception as e:
        logging.error(f"Login failed: {e}")
        await page.screenshot(path="logs/error_screenshots/login_error.png")
        raise # Re-raise error to stop the process if login fails

async def launch_and_login(p: Playwright) -> tuple[Browser | None, BrowserContext | None, Page | None]:
    """Launches browser, creates context (loading state if possible), and ensures login."""
    browser = None
    context = None
    page = None
    try:
        browser = await p.chromium.launch() # Consider adding headless=True for automation server

        # --- Authentication Strategy: Load or Login ---
        if os.path.exists(config.AUTH_STATE_FILE):
            logging.info(f"Found existing auth state: {config.AUTH_STATE_FILE}")
            context = await browser.new_context(storage_state=config.AUTH_STATE_FILE)
            page = await context.new_page()
            if await is_session_valid(page):
                logging.info("Loaded valid session from state file.")
                return browser, context, page
            else:
                logging.warning("Existing session expired or invalid. Performing new login.")
                # Close old context/page, create new ones for fresh login
                await page.close()
                await context.close()
                context = await browser.new_context() # Fresh context
                page = await context.new_page()
                if not await perform_login(page):
                    # Login failed, return None to indicate failure
                    await browser.close()
                    return None, None, None
                # If login succeeds here, page and context are valid
                return browser, context, page
        else:
            # First run scenario, no auth state exists
            logging.info("No auth state file found. Performing first login.")
            context = await browser.new_context()
            page = await context.new_page()
            if not await perform_login(page):
                # Login failed, return None
                await browser.close()
                return None, None, None
            # If login succeeds, page and context are valid
            return browser, context, page

    except Exception as e:
        logging.critical(f"Error during browser launch or login process: {e}")
        if page: await page.screenshot(path="logs/error_screenshots/launch_login_error.png")
        if context: await context.close()
        if browser: await browser.close()
        return None, None, None

# --- Course Navigation ---

async def find_and_click_course_link(page: Page, course_code: str, group_name: str | None):
    """Navigates courses page, expands group if needed, clicks specific course link."""
    logging.info(f"Attempting to navigate to course: {course_code} (Group: {group_name or 'None'})")
    await page.goto(config.COURSES_URL, wait_until="domcontentloaded") # Faster wait

    # Handle group expansion
    if group_name:
        # Use format string for XPath from config
        group_header_xpath = config.GROUP_HEADER_XPATH.format(group_name=group_name)
        try:
            group_header = page.locator(group_header_xpath).first
            await group_header.scroll_into_view_if_needed() # Ensure it's clickable
            await group_header.wait_for(state="visible", timeout=20000) # Increased timeout

            # Optional: Check if already expanded to avoid unnecessary click
            # is_expanded = await group_header.evaluate("el => el.parentElement.parentElement.classList.contains('expanded')") # Example check, adjust based on actual HTML
            # if not is_expanded:

            logging.info(f"Clicking to expand group: {group_name}")
            # Use force=True cautiously, maybe try without first
            await group_header.click(timeout=10000) # Give more time for click
            logging.info(f"Expanded group: {group_name}")
            # Wait for content within the group to potentially load/become visible
            await page.wait_for_timeout(3000) # Pause for potential dynamic loading

        except Exception as e:
            logging.error(f"Failed to find or expand group '{group_name}' for {course_code}: {e}", exc_info=True)
            await page.screenshot(path=f"logs/error_screenshots/group_error_{course_code}.png")
            raise # Re-raise error to indicate failure for this course

    # Click the specific course link using selector from config
    course_link_selector = config.COURSE_LINK_SELECTOR.format(course_code=course_code)
    try:
        course_link = page.locator(course_link_selector).first
        await course_link.wait_for(state="visible", timeout=20000) # Wait for link itself
        await course_link.scroll_into_view_if_needed()
        await course_link.click(timeout=15000) # Click the course link
        # Wait for the next page to indicate navigation started
        await page.wait_for_load_state("domcontentloaded", timeout=30000)
        logging.info(f"Successfully clicked course link for {course_code}")
    except Exception as e:
        logging.error(f"Failed to find or click course link for {course_code}: {e}", exc_info=True)
        await page.screenshot(path=f"logs/error_screenshots/course_link_error_{course_code}.png")
        raise # Re-raise error

# --- Resource Navigation (within a course page) ---

async def navigate_to_resources_section(page: Page) -> bool:
    """Clicks the Resources tab/button on a course details page."""
    logging.info("Navigating to Resources section...")
    try:
        resources_tab = page.locator(config.RESOURCES_TAB_SELECTOR).first
        await resources_tab.wait_for(state="visible", timeout=15000)
        await resources_tab.click(timeout=10000)
        # Wait for some element *within* the resources section to appear
        # Using RECORDING_ITEM_SELECTOR as a proxy, assuming recordings are common. Adjust if needed.
        await page.wait_for_selector(config.RESOURCE_ITEM_SELECTOR, state="attached", timeout=15000) # Wait for items to be in DOM
        logging.info("Successfully navigated to Resources section.")
        return True
    except Exception as e:
        logging.error(f"Failed to navigate to Resources section: {e}", exc_info=True)
        await page.screenshot(path="logs/error_screenshots/resources_nav_error.png")
        return False

async def get_all_resource_items(page: Page) -> list[Locator]:
    """Gets all resource item locators (PDFs, links, recordings) from the Resources section."""
    logging.info("Fetching all resource items from Resources section...")
    try:
        # Wait briefly to ensure items are rendered if dynamically loaded
        await page.wait_for_timeout(1500)
        items = await page.locator(config.RESOURCE_ITEM_SELECTOR).all()
        logging.info(f"Found {len(items)} resource items.")
        return items
    except Exception as e:
        logging.error(f"Failed to get resource items: {e}")
        # Optionally screenshot on error
        # await page.screenshot(path="logs/error_screenshots/get_resources_error.png")
        return [] # Return empty list on failure

# --- Deprecated/Old Navigation Logic (Keep for reference or remove) ---
# async def navigate_to_recordings_list(page: Page): ... # Replaced by navigate_to_resources_section
# async def get_all_recording_items(page: Page): ... # Replaced by get_all_resource_items