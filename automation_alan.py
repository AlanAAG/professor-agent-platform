import os
import asyncio
import argparse
import re
import requests
import time
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from openai import OpenAI
# from google import genai 

# Load environment variables from the .env file
load_dotenv()

# --- Configuration & Secrets ---
LOGIN_URL = "https://coach.tetr.com/"
COURSES_URL = LOGIN_URL + "courses"
AUTH_STATE_FILE = "auth_state.json"

USERNAME = os.environ.get("COACH_USERNAME")
PASSWORD = os.environ.get("COACH_PASSWORD")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 

# --- Playwright Selectors ---
MODAL_HEADER_SELECTOR = '.popupHeader' 
SVG_CLOSE_SELECTOR = f'{MODAL_HEADER_SELECTOR} svg' 
MODAL_CONTENT_SELECTOR = '.popupBody' # Selector for the main scrollable/visible modal body
RECORDING_LINK_SELECTOR = 'a:has-text("Click to View")' # The target link selector

# ----------------------------------------------------------------------
# --- CORE UTILITY FUNCTIONS ---
# ----------------------------------------------------------------------

# Helper Function: Session Validation
async def is_session_valid(page):
    """Checks if the saved session can successfully load the dashboard."""
    try:
        await page.goto(LOGIN_URL) 
        await page.wait_for_selector('#gtm-IdDashboard', timeout=5000)
        return True
    except Exception:
        return False

# Utility Function: Extract Drive ID
def extract_drive_file_id(url: str) -> str:
    """Extracts the file ID from a Google Drive URL."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    raise ValueError("Invalid Google Drive URL format.")

# CORE LOGIC: DOWNLOAD VIA COOKIES (Step 7 - Part 1)
def download_drive_file_with_cookies(file_id: str, cookies: list, local_filename: str):
    """
    Constructs the download URL and fetches the file using cookies from the authenticated Playwright session.
    """
    session = requests.Session()
    
    requests_cookies = {}
    for cookie in cookies:
        if 'google.com' in cookie['domain'] or '.google.com' in cookie['domain']:
            requests_cookies[cookie['name']] = cookie['value']

    session.cookies.update(requests_cookies)
    download_url = f"https://docs.google.com/uc?export=download&id={file_id}"
    
    print(f"-> Initiating authenticated download...")
    response = session.get(download_url, stream=True, allow_redirects=True)
    
    if response.status_code == 200 and 'Content-Disposition' in response.headers:
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ File successfully downloaded to {local_filename}")
        return local_filename
    
    else:
        response.raise_for_status()
        raise Exception(f"Download failed: Check file sharing/authentication. Status: {response.status_code}")

# CORE LOGIC: AUDIO EXTRACTION (Step 7 - Part 1.5)
def extract_audio_from_video(video_path: str, audio_path: str) -> str:
    raise NotImplementedError("Audio extraction requires 'moviepy' implementation.")

# CORE LOGIC: POP-UP DISMISSAL (Unblocks Clicks)
# NOTE: Pop-up functions are preserved but called only if necessary.
async def dismiss_universal_close_button(page):
    """Attempts to click the universal 'X' icon accessible on the app's header."""
    CLOSE_LOCATORS = [
        'button[aria-label="Close"]',       
        'button:has-text("Close")',
        'div[aria-label="close"]',          
        'svg[style*="cursor: pointer"]'     
    ]
    
    for selector in CLOSE_LOCATORS:
        locator = page.locator(selector).first
        try:
            if await locator.is_visible(timeout=1000): 
                await locator.click(force=True, timeout=5000)
                await locator.wait_for(state="hidden", timeout=5000) 
                print(f"— Successfully clicked and dismissed '{selector}'.")
                return True
        except Exception:
            continue
    return False

async def dismiss_all_popups(page):
    """
    Aggressively dismisses common modal overlays by prioritizing the universal 'X' icon 
    and then falling back to safe CTA clicks (like "Later").
    """
    print("-> Checking for and aggressively dismissing all overlays...")
    
    if await dismiss_universal_close_button(page):
        return

    later_locator = page.get_by_role("button", name="Later").first
    try:
        if await later_locator.is_visible(timeout=1000):
            print("— Found 'Later' button. Clicking to dismiss dialog safely.")
            await later_locator.click(force=True, timeout=5000)
            await later_locator.wait_for(state="hidden", timeout=5000)
            print("✅ 'Later' modal dismissed.")
            return
    except Exception:
        pass

    get_started_locator = page.get_by_role("button", name="Get Started").first
    try:
        if await get_started_locator.is_visible(timeout=1000):
            print("— WARNING: Clicking 'Get Started' (last resort).")
            await get_started_locator.click(force=True, timeout=5000)
            await get_started_locator.wait_for(state="hidden", timeout=5000)
            print("✅ Blockage cleared via CTA. May have triggered a subsequent modal.")
            await dismiss_all_popups(page)
            return
    except Exception:
        pass

    print("— No active overlays detected.")

# ----------------------------------------------------------------------
# --- Core Login Function (Steps 1 & 2) ---
# ----------------------------------------------------------------------
async def perform_login(page):
    """Handles the initial login process and saves authentication state."""
    print(f"-> Navigating to: {LOGIN_URL}")
    await page.goto(LOGIN_URL)

    print("-> Attempting login...")
    try:
        await page.fill('input[name="officialEmail"]', USERNAME)
        await page.fill('input[name="password"]', PASSWORD)
        await page.click('#gtmLoginStd')

        try:
            DASHBOARD_SELECTOR = '#gtm-IdDashboard'
            await page.wait_for_selector(DASHBOARD_SELECTOR, state="visible", timeout=20000)
            
            print("✅ Login Successful! Dashboard found.")
            await page.context.storage_state(path=AUTH_STATE_FILE)
            print(f"🔑 Authentication state saved to {AUTH_STATE_FILE}.")
            return True
        except Exception:
            print("❌ Login Failed: Dashboard element not found. Check credentials/network.")
            raise
    except Exception:
        raise


# ----------------------------------------------------------------------
# --- AI PROCESSING LOGIC (Steps 7, 8, 9) ---
# ----------------------------------------------------------------------
async def process_recording_file(local_video_path: str):
    """
    Handles file cleanup, transcription (Whisper), and summarization (Gemini).
    """
    print("\n\n#####################################################")
    print("# STARTING AI PROCESS (Steps 7, 8, 9) #")
    print("#####################################################")
    
    if not OPENAI_API_KEY or not GEMINI_API_KEY:
        print("🛑 ERROR: API keys are missing. Aborting AI process.")
        if os.path.exists(local_video_path): os.remove(local_video_path)
        return

    # client_openai = OpenAI(api_key=OPENAI_API_KEY)
    local_audio_path = local_video_path.replace(".mp4", ".mp3")
    
    # 1. Extract Audio (Step 7 - Part 1.5)
    try:
        # Placeholder
        final_audio_path = "temp_audio_file.mp3" 
    except Exception:
        return 

    # 2. Transcribe the audio (Step 7 - Part 2: OpenAI Whisper)
    print("-> [Step 7] Generating transcript using OpenAI Whisper...")
    
    # 3. Summarize (Offline) and Fine-tune (Step 8)
    print("-> [Step 8] Cleaning transcript and generating OFFLINE summary...")
    
    # 4. Summarize (Online/RAG) and Augment with Graphics (Step 9)
    print("-> [Step 9] Generating final summary with web search and graphics...")
    
    print("\nProcess Structure Complete! Time to implement Steps 8 & 9.")


# ----------------------------------------------------------------------
# --- Core Automation Logic (Steps 3, 4, 5, 6) ---
# ----------------------------------------------------------------------
async def search_and_download_lecture(page, context, course_name: str, lecture_title: str, lecture_date: str):
    """
    Handles navigation, subject/lecture selection, and executes the download.
    """
    local_video_path = None
    
    # Define selectors needed inside this function scope (using globals where possible)
    MODAL_CONTENT_SELECTOR = '.popupBody' 
    RECORDING_LINK_SELECTOR = 'a:has-text("Click to View")' 
    VIEW_ALL_SELECTOR = 'div.view:has-text("View All")'
    MODAL_HEADER_SELECTOR = '.popupHeader' 
    SVG_CLOSE_SELECTOR = f'{MODAL_HEADER_SELECTOR} svg' 

    try:
        # 1. Direct Navigation to Courses URL (Step 3)
        print(f"\n-> Directing browser to Subjects page: {COURSES_URL}")
        await page.goto(COURSES_URL, wait_until="networkidle") 
        
        # --- NEW STEP: Dismiss ALL Pop-ups (Crucial for unblocking Step 4) ---
        await dismiss_all_popups(page)

        # --- Step 4: Expand the Subject Panel ---
        print(f"-> Locating and expanding subject: {course_name}")
        
        subject_row_locator = page.locator('.domainHeader', has_text=course_name).first
        await subject_row_locator.wait_for(state="visible", timeout=15000)
        
        is_expanded = await subject_row_locator.evaluate("el => el.classList.contains('expanded')")
        
        if is_expanded:
            print("— Subject is already expanded. Skipping accordion click.")
        else:
            print("— Subject is collapsed. Clicking to expand.")
            await subject_row_locator.click(force=True) 
        
        lecture_card_text_locator = page.get_by_text(lecture_title, exact=True).first
        await lecture_card_text_locator.wait_for(state="visible", timeout=30000)
        
        print(f"✅ Subject '{course_name}' expanded successfully.")

        # --- Step 5: Locate and Click the Lecture Card ---
        print(f"-> Locating and clicking lecture card: {lecture_title}")
        await lecture_card_text_locator.click()
        await page.wait_for_load_state("networkidle")
        print(f"✅ Course Details page for '{lecture_title}' loaded.")
            
        # --- Click the 'Sessions' tab to render the list ---
        print("-> Activating 'Sessions' tab to display recordings list...")
        sessions_tab_locator = page.get_by_text("Sessions", exact=True)
        await sessions_tab_locator.click(timeout=10000)
        await page.wait_for_load_state("networkidle")
        print("✅ 'Sessions' tab activated.")

       print(f"\n[Step 5c] -> Locating session item for date: {lecture_date}")
        
        # 1. Locate the list item (The target container)
        date_element_xpath = f"//p[@class='date' and contains(., '{lecture_date}')]"
        session_list_item_locator = page.locator(f"{date_element_xpath}/ancestor::li").first
        
        # We need to wait for the LIST ITEM to exist before scrolling
        await session_list_item_locator.wait_for(state="attached", timeout=30000)
        
        # 2. CRITICAL FIX: Explicitly scroll the element into view.
        # This resolves issues where the item is rendered but hidden by a scroll mask.
        print("— Forcing element into viewport via scroll...")
        await session_list_item_locator.scroll_into_view_if_needed()
        
        # 3. Wait for visual stability before clicking
        await session_list_item_locator.wait_for(state="visible", timeout=5000) 

        # 4. Find the specific collapse/expand area within the item and click
        print("— Attempting click on session item to expand details...")
        
        # Assume the click is the expansion action itself
        try:
            # Try clicking the entire list item aggressively
            await session_list_item_locator.click(force=True, timeout=5000) 
        except Exception as e:
            # If the click still fails (due to overlay), log and fail
            print(f"❌ Click on session list item failed. Details: {e}")
            raise Exception("Failed to click and expand session item.")

        # 5. Wait for the content to fully expand (wait for "View All" to become visible after the click)
        VIEW_ALL_SELECTOR = 'div.view:has-text("View All")'
        await page.wait_for_selector(VIEW_ALL_SELECTOR, timeout=10000) 
        print(f"✅ Session for {lecture_date} successfully expanded.")
        
        # --- CRITICAL FIX 1: Robust Expansion Click ---
        # Find the arrow icon or click the entire item aggressively
        try:
            expansion_arrow_locator = session_list_item.locator("div.expand-icon, svg, path").first 
            await expansion_arrow_locator.click(force=True, timeout=5000)
            print("— Clicked specific expansion arrow.")
        except Exception:
            await session_list_item.click(force=True, timeout=5000) 
            print("— Clicked entire list item.")
        
        # 2. Click "View All" to open modal
        view_all_locator = page.locator(VIEW_ALL_SELECTOR)
        await view_all_locator.wait_for(state="visible", timeout=10000)
        await view_all_locator.click(force=True, timeout=15000) 
        
        # 3. Wait for modal to load and stabilize
        print("-> Waiting for recording modal to stabilize...")
        await page.wait_for_selector(MODAL_HEADER_SELECTOR, timeout=30000) 
        await page.wait_for_selector('h5:has-text("Session Recording")', timeout=10000) 

        # -----------------------------------------------------------------------
        # --- CRITICAL FIX 2: Scoped and Robust Click Implementation ---
        # -----------------------------------------------------------------------
        
        # Define the link locator, SCOPED TO THE MODAL CONTENT (prevents background clicks)
        recording_link_locator = page.locator(f'{MODAL_CONTENT_SELECTOR} {RECORDING_LINK_SELECTOR}').first
        await recording_link_locator.wait_for(state="visible", timeout=10000) 

        print("-> Executing robust click to open Google Drive tab...")
        
        # 4. Use page.context.expect_page (Playwright's best method for new windows/tabs)
        async with page.context.expect_page() as new_page_info:
            try:
                # Use a single, targeted click method
                await recording_link_locator.click(force=True, timeout=10000)
            except Exception as e:
                # Fallback: Click the underlying element handle if the locator click fails
                element_handle = await recording_link_locator.element_handle()
                if element_handle:
                    await element_handle.click(force=True, timeout=10000)
                else:
                    raise Exception(f"Could not execute click on recording link. Error: {e}")

        # Capture the new page context
        recording_page = await new_page_info.value
        await recording_page.wait_for_load_state("load")
        
        # CRITICAL CHECK: Ensure the URL actually leads to Google Drive
        recording_url = recording_page.url
        if "drive.google.com" not in recording_url:
            raise Exception(f"Click failed to open Google Drive link. Final URL: {recording_url}")
            
        print(f"✅ Recording link clicked! New URL (Google Drive): {recording_url}")
        
        # --- EXECUTE DOWNLOAD (Step 7 - Part 1) ---
        file_id = extract_drive_file_id(recording_url)
        
        # Fetch cookies *after* authentication and *before* download attempt
        google_drive_cookies = await context.cookies() 
        
        local_video_path = f"lecture_{file_id}.mp4"
        download_drive_file_with_cookies(file_id, google_drive_cookies, local_video_path)
        
        # 5. Cleanup
        await recording_page.close() # Close the new Google Drive tab
        await page.click(SVG_CLOSE_SELECTOR) # Close the original modal
        
        # --- PROCEED TO AI STEPS (7, 8, 9) ---
        if local_video_path and os.path.exists(local_video_path):
            await process_recording_file(local_video_path)

    except Exception as e:
        # --- GLOBAL ERROR HANDLER ---
        error_path = f"error_screenshot_{int(time.time())}.png"
        await page.screenshot(path=error_path)
        print("\n[CRITICAL ERROR DETECTED IN NAVIGATION/DOWNLOAD STAGE]")
        print(f"❌ Automation failed. Screenshot saved to {error_path}.")
        print(f"Details: {e}")
        # Clean up video file if it was partially downloaded
        if local_video_path and os.path.exists(local_video_path):
            os.remove(local_video_path)
        return


# ----------------------------------------------------------------------
# --- Main Execution (Handles User Arguments) ---
# ----------------------------------------------------------------------
async def main():
    # --- Setup Argument Parser ---
    parser = argparse.ArgumentParser(description="Automate fetching and summarizing a Coach lecture.")
    parser.add_argument('--subject', required=True, type=str, 
                        help='The exact title of the subject (e.g., "Quantitative Tools for Business").')
    parser.add_argument('--lecture', required=True, type=str, 
                        help='The exact title of the lecture card (e.g., "How to identify gaps in the market").')
    parser.add_argument('--date', required=True, type=str, 
                        help='The exact date of the session (e.g., "08/09/2025").')
    
    args = parser.parse_args()
    
    if not USERNAME or not PASSWORD:
        print("ERROR: COACH_USERNAME or COACH_PASSWORD not set in .env or environment variables.")
        return
    
    # Arguments mapped to variables
    TARGET_COURSE = args.subject
    TARGET_LECTURE_TITLE = args.lecture
    TARGET_LECTURE_DATE = args.date
    
    print(f"\n🚀 Starting automation with inputs:")
    print(f"   Subject: {TARGET_COURSE}")
    print(f"   Lecture: {TARGET_LECTURE_TITLE}")
    print(f"   Date:    {TARGET_LECTURE_DATE}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = None
        
        # --- Authentication Strategy: Load or Login ---
        if os.path.exists(AUTH_STATE_FILE):
            context = await browser.new_context(storage_state=AUTH_STATE_FILE)
            page = await context.new_page()
            
            if await is_session_valid(page):
                print("🔑 Loaded existing authenticated session. Skipping login.")
            else:
                print("❌ Session expired. Re-authenticating.")
                context = await browser.new_context()
                page = await context.new_page()
                if not await perform_login(page):
                    await browser.close()
                    return
        else:
            # First run scenario
            context = await browser.new_context()
            page = await context.new_page()
            if not await perform_login(page):
                await browser.close()
                return
        
        # --- Run the main processing logic (Steps 3, 4, 5, 6) ---
        await search_and_download_lecture(page, context, TARGET_COURSE, TARGET_LECTURE_TITLE, TARGET_LECTURE_DATE)
        
        await browser.close()

if __name__ == "__main__":
    import time
    asyncio.run(main())