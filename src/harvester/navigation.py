# src/harvester/navigation.py

import logging
import os  # Added for AUTH_STATE_FILE path
import re
import random
import tempfile
from playwright.async_api import Page, Locator, Playwright, Browser, BrowserContext
from playwright_stealth import stealth_async
from . import config  # Imports selectors and URLs from config.py
# Assuming utils.py is in src/shared/
from src.shared import utils  # For utility functions if needed later

# A small pool of modern desktop user agents to rotate between.
USER_AGENTS = [
    # Windows 10/11 Chrome
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    # macOS Sonoma Chrome
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    # Linux Chrome
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
]

# --- Authentication & Initialization ---

async def is_session_valid(page: Page) -> bool:
    """Checks if the saved session can successfully load the COURSES page AND render content."""
    logging.info("Checking if session is valid by navigating to Courses page...")
    try:
        await page.goto(config.COURSES_URL, wait_until="domcontentloaded", timeout=30000)
        # Allow redirects or blocking JS to potentially trigger
        await page.wait_for_timeout(2000)

        current_url = page.url or ""
        if "/login" in current_url:
            logging.warning("Session invalid: Redirected back to /login.")
            return False

        # --- Stricter: verify visible content on the courses page ---
        first_default_course_code = list(config.DEFAULT_VISIBLE_COURSES)[0]
        visible_content_selector = config.COURSE_LINK_SELECTOR.format(course_code=first_default_course_code)
        logging.info(f"Checking for visible content using selector: {visible_content_selector}")
        await page.locator(visible_content_selector).first.wait_for(state="visible", timeout=20000)

        logging.info("Session is valid: Successfully loaded /courses page and verified visible content.")
        return True

    except Exception as e:
        logging.warning(f"Session validation failed: {e}")
        try:
            os.makedirs("logs/error_screenshots", exist_ok=True)
            await page.screenshot(path="logs/error_screenshots/session_validation_fail.png")
            page_content = await page.content()
            logging.warning(f"Validation failure page content (first 500 chars):\n{page_content[:500]}")
        except Exception as screen_err:
            logging.error(f"Failed to capture screenshot/content on validation failure: {screen_err}")
        return False

async def perform_login(page: Page):
    """Handles the initial login process."""
    # Ensure directories for artifacts exist to avoid I/O errors
    os.makedirs("logs/error_screenshots", exist_ok=True)
    os.makedirs(os.path.dirname(config.AUTH_STATE_FILE) or ".", exist_ok=True)

    logging.info(f"Navigating to login page: {config.LOGIN_URL}")
    try:
        # Use extended timeout and wait for DOM content; some sites need longer to initialize
        await page.goto(config.LOGIN_URL, wait_until="domcontentloaded", timeout=60000)
        logging.info("Initial navigation complete. Waiting for potential dynamic loading...")
        # Give the page extra time to execute any client-side rendering code
        await page.wait_for_timeout(5000)
    except Exception as goto_err:
        logging.error(f"Failed even during page.goto: {goto_err}")
        # Try to capture immediate context for diagnostics
        try:
            await page.screenshot(path="logs/error_screenshots/goto_login_error.png")
        except Exception:
            pass
        raise

    # Capture a screenshot right after the initial wait to observe render state
    try:
        await page.screenshot(path="logs/error_screenshots/after_login_goto.png")
        logging.info("Saved screenshot after page.goto and initial wait.")
    except Exception as screen_err:
        logging.error(f"Failed to save post-goto screenshot: {screen_err}")

    # Log the received HTML to help diagnose blank or blocked pages
    try:
        page_content = await page.content()
        logging.info(f"Page content length after load: {len(page_content)} chars")
        logging.info(f"Page content start:\n{page_content[:500]}")
    except Exception as content_err:
        logging.error(f"Failed to get page content: {content_err}")

    # Best-effort: allow network to go idle, but do not require it
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
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
    """Launch a persistent browser context with stealth and ensure a valid session.

    Returns (browser, context, page) where browser is always None when using
    launch_persistent_context. Callers should only close the context.
    """
    browser: Browser | None = None  # Using persistent context, browser object is not returned
    context: BrowserContext | None = None
    page: Page | None = None

    # Persistent user data directory (env override supports experimentation)
    user_data_dir = os.environ.get("PW_USER_DATA_DIR") or os.path.join("data", "pw_user_data")
    os.makedirs(user_data_dir, exist_ok=True)
    logging.info(f"Using persistent user data directory: {user_data_dir}")

    # Engine and headless toggles via environment
    engine_name = (os.environ.get("PW_ENGINE") or "chromium").strip().lower()
    headless_env = (os.environ.get("PW_HEADLESS") or "true").strip().lower()
    headless = headless_env in ("1", "true", "yes")

    try:
        # Select browser engine
        if engine_name == "firefox":
            engine = p.firefox
        elif engine_name == "webkit":
            engine = p.webkit
        else:
            engine = p.chromium
            engine_name = "chromium"

        # Build common launch options
        stealth_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--window-size=1440,900",
            "--lang=en-US,en",
            "--accept-lang=en-US,en;q=0.9",
        ]

        launch_kwargs: dict = {
            "headless": headless,
            "user_agent": random.choice(USER_AGENTS),
            "viewport": {"width": 1440, "height": 900},
            "ignore_https_errors": True,
        }

        # 'args' is primarily used by Chromium; include only when launching Chromium to avoid warnings
        if engine_name == "chromium":
            launch_kwargs["args"] = stealth_args

        # Launch persistent context
        context = await engine.launch_persistent_context(user_data_dir, **launch_kwargs)

        # Add init script after context is created to hide webdriver
        await context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        # Get or create a page
        page = context.pages[0] if context.pages else await context.new_page()

        # Log browser console messages for debugging detection issues
        try:
            page.on(
                "console",
                lambda msg: logging.warning(f"BROWSER CONSOLE [{msg.type}]: {msg.text}")
            )
        except Exception:
            # Best-effort; do not fail launch if event hook fails
            pass

        # Apply stealth AFTER the page exists
        await stealth_async(page)

        # Try using the existing persisted session
        if await is_session_valid(page):
            logging.info("Session from persistent context is valid.")
            return None, context, page
        else:
            logging.warning("Session from persistent context invalid or not found. Performing login...")

            if await perform_login(page):
                logging.info("Login successful within persistent context.")
                # Optionally clear cookies after a forced login to drop any stale state
                try:
                    await context.clear_cookies()
                    logging.info("Cleared old cookies after forced login.")
                except Exception as clear_err:
                    logging.warning(f"Could not clear context storage after forced login: {clear_err}")

                # Re-validate after login
                if await is_session_valid(page):
                    return None, context, page
                else:
                    raise RuntimeError("Session still invalid after forced login.")
            else:
                logging.error("Login failed even with persistent context.")
                await context.close()
                return None, None, None

    except Exception as e:
        logging.critical(f"Error during persistent context launch or login process: {e}", exc_info=True)
        try:
            if page:
                await page.screenshot(path="logs/error_screenshots/launch_login_error.png")
        except Exception:
            logging.error("Failed to write launch/login error screenshot.")
        if context:
            try:
                await context.close()
            except Exception:
                pass
        return None, None, None

# --- Course Navigation ---

async def find_and_click_course_link(page: Page, course_code: str, group_name: str | None):
    """Navigates courses page and clicks the target course link.
    Mirrors partner Selenium logic: if course is default-visible, click directly;
    otherwise expand the owning group first, then click the course link.
    """
    logging.info(f"Attempting to navigate to course: {course_code} (Group hint: {group_name or 'None'})")
    await page.goto(config.COURSES_URL, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(3000)  # Allow time for JS to potentially run/fail

    # If we were redirected to login, fail fast with a clear error
    if "/login" in (page.url or ""):
        raise RuntimeError("Not authenticated: redirected to login when opening courses page.")

    # --- NEW: Check for Blank Page / Force Relogin ---
    is_blank = False
    try:
        first_default_course_code = list(config.DEFAULT_VISIBLE_COURSES)[0]
        check_selector = config.COURSE_LINK_SELECTOR.format(course_code=first_default_course_code)
        await page.locator(check_selector).first.wait_for(state="visible", timeout=10000)
        logging.info("Courses page content seems visible.")
    except Exception:
        logging.warning("Courses page appears blank or key content missing after navigation.")
        is_blank = True
        # Capture state
        try:
            os.makedirs("logs/error_screenshots", exist_ok=True)
            await page.screenshot(path=f"logs/error_screenshots/blank_courses_page_{course_code}.png")
            page_content = await page.content()
            logging.warning(f"Blank page content (first 500 chars):\n{page_content[:500]}")
        except Exception:
            pass

        # Attempt re-login
        logging.warning("Attempting force re-login within the current browser...")
        try:
            login_success = await perform_login(page)
            if not login_success:
                raise RuntimeError("Forced re-login failed.")
            logging.info("Forced re-login successful. Re-navigating to courses...")
            await page.goto(config.COURSES_URL, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(3000)
            await page.locator(check_selector).first.wait_for(state="visible", timeout=15000)
            logging.info("Courses page loaded successfully after forced re-login.")
        except Exception as relogin_err:
            logging.error(f"Failed to recover session with forced re-login: {relogin_err}")
            raise RuntimeError(f"Could not load courses page even after re-login attempt: {relogin_err}") from relogin_err
    # --- END CHECK ---

    # --- ADD DEBUG SCREENSHOT AND CONTENT LOG ---
    try:
        os.makedirs("logs/error_screenshots", exist_ok=True)
        logging.info(f"Attempting to save screenshot of courses page: {page.url}")
        await page.screenshot(path=f"logs/error_screenshots/courses_page_load_{course_code}.png")
        page_content = await page.content()
        logging.info(f"Courses page content length: {len(page_content)} chars")
        logging.info(f"Courses page content start:\n{page_content[:500]}")
        # Optionally save full HTML for deeper inspection
        # with open(f"logs/courses_page_{course_code}.html", "w", encoding="utf-8") as f:
        #     f.write(page_content)
    except Exception as screen_err:
        logging.error(f"Failed to save courses page debug info: {screen_err}")
    # --- END DEBUG ---

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