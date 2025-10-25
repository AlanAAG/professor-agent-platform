# src/harvester/navigation.py

import logging
import os  # Added for AUTH_STATE_FILE path
import re
from playwright.async_api import Page, Locator, Playwright, Browser, BrowserContext
from playwright_stealth import stealth_async
from . import config  # Imports selectors and URLs from config.py
# Assuming utils.py is in src/shared/
from src.shared import utils  # For utility functions if needed later

# --- Authentication & Initialization ---

async def is_session_valid(page: Page) -> bool:
    """Checks if the saved session can successfully load the dashboard."""
    logging.info("Checking if session is valid...")
    try:
        await page.goto(config.BASE_URL, wait_until="domcontentloaded", timeout=20000)
        # Allow any SPA redirects to settle
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            pass

        # If redirected back to login, session is not valid
        if "/login" in (page.url or ""):  # Be robust to None
            logging.info("Session invalid: redirected to /login")
            return False

        # Prefer presence of a known dashboard element; otherwise accept non-login URL as valid
        try:
            await page.wait_for_selector(config.DASHBOARD_INDICATOR, state="visible", timeout=8000)
            logging.info("Session is valid (dashboard indicator visible).")
            return True
        except Exception:
            logging.info("Dashboard indicator not found, but not on /login. Treating session as valid.")
            return True
    except Exception as e:
        logging.warning(f"Session validation failed: {e}")
        return False

async def perform_login(page: Page):
    """Handles the initial login process."""
    # Ensure directories for artifacts exist to avoid I/O errors
    os.makedirs("logs/error_screenshots", exist_ok=True)
    os.makedirs(os.path.dirname(config.AUTH_STATE_FILE) or ".", exist_ok=True)

    logging.info(f"Navigating to login page: {config.LOGIN_URL}")
    await page.goto(config.LOGIN_URL, wait_until="domcontentloaded")
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        # Not all pages reach networkidle; continue
        pass

    logging.info("Attempting login...")
    username = os.environ.get("COACH_USERNAME")
    password = os.environ.get("COACH_PASSWORD")

    if not username or not password:
        logging.critical("COACH_USERNAME or COACH_PASSWORD not set in environment. Cannot login.")
        raise ValueError("Missing login credentials in environment variables.")

    try:
        # Ensure login form is visible before interacting
        await page.wait_for_selector(config.USERNAME_SELECTOR, state="visible", timeout=20000)
        await page.wait_for_selector(config.PASSWORD_SELECTOR, state="visible", timeout=20000)
        # Fill credentials and submit
        await page.locator(config.USERNAME_SELECTOR).fill(username)
        await page.locator(config.PASSWORD_SELECTOR).fill(password)
        await page.locator(config.LOGIN_BUTTON_SELECTOR).click()

        # Wait for redirect away from the login page OR for dashboard indicator
        login_url_patterns = [
            re.compile(r"/login/?$"),
        ]
        try:
            await page.wait_for_url(re.compile(r"/(courses|dashboard)"), timeout=30000)
        except Exception:
            # Fallback to dashboard indicator
            try:
                await page.wait_for_selector(config.DASHBOARD_INDICATOR, state="visible", timeout=15000)
            except Exception as dash_err:
                # If still on login URL, treat as failure
                current_url = page.url
                if any(p.search(current_url or "") for p in login_url_patterns):
                    raise dash_err
                # Otherwise continue; some tenants may land elsewhere (e.g., /courses)

        # Final sanity: do not remain on /login
        if "/login" in (page.url or ""):
            raise RuntimeError("Login appears to have failed; still on /login after submit.")

        logging.info("Login successful!")
        # Save authentication state
        await page.context.storage_state(path=config.AUTH_STATE_FILE)
        logging.info(f"Authentication state saved to {config.AUTH_STATE_FILE}")
        return True
    except Exception as e:
        logging.error(f"Login failed: {e}")
        try:
            # Allow UI to render before screenshot to avoid all-white captures
            try:
                await page.wait_for_timeout(1500)
            except Exception:
                pass
            await page.screenshot(path="logs/error_screenshots/login_error.png")
            # Also save HTML to aid debugging
            try:
                html_content = await page.content()
                with open("logs/error_screenshots/login_error.html", "w", encoding="utf-8") as f:
                    f.write(html_content)
            except Exception:
                pass
        except Exception:
            logging.error("Failed to write login error screenshot.")
        raise # Re-raise error to stop the process if login fails

async def launch_and_login(p: Playwright) -> tuple[Browser | None, BrowserContext | None, Page | None]:
    """Launches browser, creates context (loading state if possible), and ensures login."""
    browser = None
    context = None
    page = None

    # Use Playwright's default modern User-Agent (more compatible than hardcoding an older UA)

    try:
        browser = await p.chromium.launch()

        if os.path.exists(config.AUTH_STATE_FILE):
            logging.info(f"Found existing auth state: {config.AUTH_STATE_FILE}")
            context = await browser.new_context(storage_state=config.AUTH_STATE_FILE)
            page = await context.new_page()
            await stealth_async(page)  # <-- APPLY STEALTH

            if await is_session_valid(page):
                logging.info("Loaded valid session from state file.")
                return browser, context, page
            else:
                logging.warning("Existing session expired or invalid. Performing new login.")
                await page.close()
                await context.close()

                context = await browser.new_context()
                page = await context.new_page()
                await stealth_async(page)  # <-- APPLY STEALTH

                if not await perform_login(page):
                    await browser.close()
                    return None, None, None
                return browser, context, page
        else:
            logging.info("No auth state file found. Performing first login.")
            context = await browser.new_context()
            page = await context.new_page()
            await stealth_async(page)  # <-- APPLY STEALTH

            if not await perform_login(page):
                await browser.close()
                return None, None, None
            return browser, context, page

    except Exception as e:
        logging.critical(f"Error during browser launch or login process: {e}")
        try:
            if page:
                await page.screenshot(path="logs/error_screenshots/launch_login_error.png")
        except Exception:
            logging.error("Failed to write launch/login error screenshot.")
        if context:
            await context.close()
        if browser:
            await browser.close()
        return None, None, None

# --- Course Navigation ---

async def find_and_click_course_link(page: Page, course_code: str, group_name: str | None):
    """Navigates courses page and clicks the target course link.
    Mirrors partner Selenium logic: if course is default-visible, click directly;
    otherwise expand the owning group first, then click the course link.
    """
    logging.info(f"Attempting to navigate to course: {course_code} (Group hint: {group_name or 'None'})")
    await page.goto(config.COURSES_URL, wait_until="domcontentloaded")
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass

    # If we were redirected to login, fail fast with a clear error
    if "/login" in (page.url or ""):
        raise RuntimeError("Not authenticated: redirected to login when opening courses page.")

    try:
        if course_code in config.DEFAULT_VISIBLE_COURSES:
            logging.info(f"Course {course_code} is default-visible. Searching directly...")
            target_selector = config.COURSE_LINK_SELECTOR.format(course_code=course_code)
            course_link = page.locator(target_selector).first
            await course_link.wait_for(state="visible", timeout=20000)
            await course_link.scroll_into_view_if_needed()
            await page.wait_for_timeout(500)
            await course_link.click(timeout=15000)
            await page.wait_for_load_state("domcontentloaded", timeout=30000)
            logging.info(f"Successfully opened course {course_code}")
            return

        # Non-default visible: expand group then click
        effective_group = group_name or (config.COURSE_MAP.get(course_code, {}).get("group"))
        if not effective_group:
            logging.warning(f"Group for course {course_code} is not defined. Attempting direct link click anyway.")
            target_selector = config.COURSE_LINK_SELECTOR.format(course_code=course_code)
            course_link = page.locator(target_selector).first
            await course_link.wait_for(state="visible", timeout=20000)
            await course_link.scroll_into_view_if_needed()
            await course_link.click(timeout=15000)
            await page.wait_for_load_state("domcontentloaded", timeout=30000)
            logging.info(f"Successfully opened course {course_code}")
            return

        logging.info(f"Expanding group '{effective_group}' for course {course_code}...")
        group_header_selector = config.GROUP_HEADER_SELECTOR.format(group_name=effective_group)
        group_header = page.locator(group_header_selector).first
        await group_header.wait_for(state="visible", timeout=20000)
        await group_header.scroll_into_view_if_needed()
        await page.wait_for_timeout(500)
        await group_header.click(timeout=10000)
        logging.info(f"Clicked group '{effective_group}'. Waiting for course link to appear...")

        target_selector = config.COURSE_LINK_SELECTOR.format(course_code=course_code)
        # Wait for the course link to be present in the DOM, then visible (expanded)
        await page.wait_for_selector(target_selector, state="attached", timeout=20000)
        course_link = page.locator(target_selector).first
        await course_link.wait_for(state="visible", timeout=20000)
        await course_link.scroll_into_view_if_needed()
        await course_link.click(timeout=15000)
        await page.wait_for_load_state("domcontentloaded", timeout=30000)
        logging.info(f"Successfully opened course {course_code}")

    except Exception as e:
        logging.error(f"Course navigation failed for {course_code}: {e}", exc_info=True)
        await page.screenshot(path=f"logs/error_screenshots/course_nav_error_{course_code}.png")
        raise

# --- Resource Navigation (within a course page) ---

async def navigate_to_resources_section(page: Page) -> bool:
    """Clicks the Resources tab/button on a course details page using robust waits."""
    logging.info("Navigating to Resources section...")
    try:
        # Wait for the page to be mostly loaded
        await page.wait_for_load_state("networkidle", timeout=20000)

        resources_tab = page.locator(config.RESOURCES_TAB_SELECTOR).first

        logging.info(f"Waiting for selector: '{config.RESOURCES_TAB_SELECTOR}'")
        # Wait for the element to be visible
        await resources_tab.wait_for(state="visible", timeout=25000)
        # Wait for it to be enabled (ElementHandle state for 'enabled')
        handle = await resources_tab.element_handle()
        if handle is not None:
            await handle.wait_for_element_state("enabled", timeout=5000)

        logging.info("Resources tab found and enabled. Attempting click...")
        await resources_tab.click(timeout=10000)

        # Wait for ANY of the potential section headers to appear to confirm content loaded
        logging.info("Waiting for resource section headers to appear...")
        any_section_header_selector = (
            f"{config.PRE_READ_SECTION_SELECTOR},"
            f"{config.IN_CLASS_SECTION_SELECTOR},"
            f"{config.POST_CLASS_SECTION_SELECTOR},"
            f"{config.RECORDINGS_LINK_SELECTOR}"
        )
        # Wait for the first visible element matching any of these selectors
        await page.locator(any_section_header_selector).first.wait_for(state="visible", timeout=20000)
        logging.info("Successfully navigated to Resources section and section headers are visible.")
        return True
    except Exception as e:
        logging.error(f"Failed to navigate to Resources section: {e}", exc_info=True)
        # ... (screenshot logic) ...
        try:
            await page.screenshot(path="logs/error_screenshots/resources_nav_error.png")
        except Exception:
            pass
        return False

async def get_all_resource_items(page: Page) -> list[Locator]:
    """Gets all resource item locators (PDFs, links, recordings) from the Resources section."""
    logging.info("Fetching all resource items from Resources section...")
    try:
        # Wait briefly to ensure items are rendered if dynamically loaded
        await page.wait_for_timeout(1500)
        locator = page.locator(config.RESOURCE_ITEM_SELECTOR)
        count = await locator.count()
        if count == 0:
            logging.warning("No resource items found with primary selector. Trying fallback on anchors within fileBox.")
            # Fallback: return anchors inside fileBox to at least extract URLs
            locator = page.locator(f"{config.RESOURCE_ITEM_SELECTOR} a")
            count = await locator.count()
        items = [locator.nth(i) for i in range(count)]
        logging.info(f"Found {count} resource items.")
        return items
    except Exception as e:
        logging.error(f"Failed to get resource items: {e}")
        # Optionally screenshot on error
        # await page.screenshot(path="logs/error_screenshots/get_resources_error.png")
        return [] # Return empty list on failure