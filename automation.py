import os
import asyncio
import argparse
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Load environment variables from the .env file
# Loads COACH_USERNAME, COACH_PASSWORD, and GEMINI_API_KEY
load_dotenv()

# --- Configuration ---
LOGIN_URL = "https://coach.tetr.com/"
COURSES_URL = LOGIN_URL + "courses"
AUTH_STATE_FILE = "auth_state.json"

# Retrieve credentials securely from environment variables
USERNAME = os.environ.get("COACH_USERNAME")
PASSWORD = os.environ.get("COACH_PASSWORD")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Helper Function: Session Validation ---
async def is_session_valid(page):
    """Checks if the saved session can successfully load the dashboard."""
    try:
        await page.goto(LOGIN_URL) 
        await page.wait_for_selector('#gtm-IdDashboard', timeout=5000)
        return True
    except Exception:
        return False

# ----------------------------------------------------------------------
# --- Core Login Function (Steps 1 & 2) ---
# ----------------------------------------------------------------------
async def perform_login(page):
    """Handles the initial login process and saves authentication state."""
    print(f"-> Navigating to: {LOGIN_URL}")
    await page.goto(LOGIN_URL)

    print("-> Attempting login...")
    try:
        # Confirmed selectors for login inputs
        await page.fill('input[name="officialEmail"]', USERNAME)
        await page.fill('input[name="password"]', PASSWORD)
        
        # Confirmed selector for login button
        await page.click('#gtmLoginStd')

        # --- Dashboard/Post-Login Check ---
        try:
            DASHBOARD_SELECTOR = '#gtm-IdDashboard'
            await page.wait_for_selector(DASHBOARD_SELECTOR, state="visible", timeout=20000)
            
            print("✅ Login Successful! Dashboard found.")
            await page.context.storage_state(path=AUTH_STATE_FILE)
            print(f"🔑 Authentication state saved to {AUTH_STATE_FILE}.")
            return True

        except Exception:
            print("❌ Login Failed: Dashboard element not found. Check credentials/network.")
            await page.screenshot(path="login_error_dashboard_fail.png")
            return False

    except Exception as e:
        print(f"❌ Initial Login Form Failed (Timeout or Selector Error): {e}")
        await page.screenshot(path="login_error_form_fail.png")
        return False

# ----------------------------------------------------------------------
# --- Core Automation Logic (Steps 3, 4, 5, 6) ---
# ----------------------------------------------------------------------
async def search_and_download_lecture(page, context, course_name: str, lecture_title: str, lecture_date: str):
    """
    Handles navigation, subject/lecture selection, and extracts the Google Drive URL.
    """
    # 1. Direct Navigation to Courses URL (Step 3)
    print(f"\n-> Directing browser to Subjects page: {COURSES_URL}")
    await page.goto(COURSES_URL, wait_until="networkidle") 
    
    # --- Step 4: Expand the Subject Panel ---
    print(f"-> Locating and expanding subject: {course_name}")
    try:
        subject_panel = page.get_by_text(course_name, exact=True).first
        await subject_panel.click()
        await page.wait_for_timeout(1000) 
        print(f"✅ Subject '{course_name}' expanded successfully.")
    except Exception:
        print(f"❌ Error in Step 4: Could not find or click subject panel with text '{course_name}'.")
        return 

    # --- Step 5: Locate and Click the Lecture Card ---
    print(f"-> Locating and clicking lecture card: {lecture_title}")
    try:
        # Locate the lecture card by its title text
        lecture_card_link = page.get_by_text(lecture_title, exact=True).first
        await lecture_card_link.wait_for(state="visible", timeout=15000) 
        await lecture_card_link.click()
        await page.wait_for_load_state("networkidle")
        print(f"✅ Course Details page for '{lecture_title}' loaded.")
        
    except Exception as e:
        print(f"❌ Error in Step 5: Could not find or click lecture card with title '{lecture_title}': {e}")
        await page.screenshot(path="lecture_select_error.png")
        return
        
    # --- Step 5b: Click the "Sessions" link on the Details Page ---
    print("-> Clicking 'Sessions' to view recordings list...")
    try:
        # Locate and click the 'Sessions' element (based on text)
        sessions_locator = page.get_by_text("Sessions", exact=True)
        await sessions_locator.wait_for(state="visible", timeout=15000)
        await sessions_locator.click()
        await page.wait_for_load_state("networkidle")
        print("✅ Sessions list page loaded.")
        
    except Exception as e:
        print(f"❌ Error in Step 5b: Could not find or click the 'Sessions' element: {e}")
        await page.screenshot(path="sessions_click_error.png")
        return

    # --- Step 5c: Locate and Expand the specific Session by Date (FINALIZED) ---
    print(f"\n[Step 5c] -> Locating session item for date: {lecture_date}")
    
    # Selectors
    VIEW_ALL_SELECTOR = 'div.view:has-text("View All")'
    MODAL_HEADER_SELECTOR = '.popupHeader' 
    
    try:
        # FIX: Wait for the session list to render before trying to locate a specific item
        await page.wait_for_selector('li:has-text("PM")', timeout=15000) 
        print("[LOG 0/5] Session list content detected.")
        
        # 1. Locate the date element (p.date) and target the stable <li> ancestor
        date_element_xpath = f"//p[@class='date' and contains(., '{lecture_date}')]"
        session_list_item = page.locator(f'{date_element_xpath}/ancestor::li').first

        if not await session_list_item.is_visible():
            print(f"[ERROR LOG] Could not find the stable list item (<li>) ancestor for date '{lecture_date}'.")
            raise Exception("Session list item not found via ancestor search.")

        print(f"[LOG 1/5] Session list item found. Attempting initial click...")
        
        # 2. Click the session list item header to expand the immediate details (uses force=True)
        await session_list_item.click(force=True, timeout=10000) 
        print(f"[LOG 2/5] Initial click on session item succeeded. Waiting for 'View All'...")
        
        # 3. Wait for the "View All" button to appear 
        view_all_locator = page.locator(VIEW_ALL_SELECTOR)
        await view_all_locator.wait_for(state="visible", timeout=10000)
        print(f"[LOG 3/5] 'View All' button found and is visible. Attempting second click...")

        # 4. Click the "View All" button to open the final session modal (using force=True)
        await view_all_locator.click(force=True, timeout=15000) 
        print(f"[LOG 4/5] 'View All' click succeeded. Waiting for modal element...")

        # 5. Wait for the MODAL HEADER to be visible (Confirms modal is loaded)
        await page.wait_for_selector(MODAL_HEADER_SELECTOR, timeout=30000) 
        
        # 6. Now that the modal is loaded, wait for the target content text (H5)
        await page.wait_for_selector('h5:has-text("Session Recording")', timeout=10000) 

        print(f"[LOG 5/5] Final modal loaded successfully!")
        print(f"✅ Session details modal for '{lecture_date}' loaded.")
        
    except Exception as e:
        print("\n[CRITICAL FAILURE DETECTED IN STEP 5c]")
        print(f"❌ Error in Step 5c: Automation stopped. Details: {e}")
        await page.screenshot(path="session_expand_error.png")
        return
        
    # --- Step 6: Get the Recording Link ---
    RECORDING_LINK_SELECTOR = 'a:has-text("Click to View")'
    print(f"-> Clicking the final recording link ({RECORDING_LINK_SELECTOR})...")
    
    # The modal close button is an SVG inside the MODAL_HEADER_SELECTOR
    SVG_CLOSE_SELECTOR = f'{MODAL_HEADER_SELECTOR} svg' 
    
    recording_url = None
    try:
        # Locate the clickable link using only its unique text
        recording_link_locator = page.get_by_text("Click to View", exact=True).first
        
        # Explicitly wait for it to be visible before clicking
        await recording_link_locator.wait_for(state="visible", timeout=10000) 

        # Expect a new tab/page to open (Google Drive)
        async with context.expect_page() as new_page_info:
            # CRITICAL FIX: Use force=True to bypass scrolling/viewport checks and guarantee the click
            await recording_link_locator.click(force=True)
        
        recording_page = await new_page_info.value
        await recording_page.wait_for_load_state("load")
        
        recording_url = recording_page.url
        print(f"✅ Recording link clicked! New URL (Google Drive): {recording_url}")
        
        # Close the modal ('X' button)
        await page.click(SVG_CLOSE_SELECTOR) 
        
        # Close the new tab
        await recording_page.close()
        
    except Exception as e:
        print(f"❌ Error in Step 6: Could not click 'Click to View' or open new tab: {e}")
        await page.screenshot(path="recording_link_error.png")
        return
        
    # --- PROCEED TO AI STEPS (7, 8, 9) ---
    if recording_url:
        await process_recording_file(recording_url)

# ----------------------------------------------------------------------
# --- AI PROCESSING LOGIC (Steps 7, 8, 9) - To be Implemented ---
# ----------------------------------------------------------------------
async def process_recording_file(recording_url: str):
    """
    Handles file download, transcription, and summarization (Steps 7, 8, 9).
    """
    print("\n\n#####################################################")
    print("# STARTING AI PROCESS (Steps 7, 8, 9) #")
    print("#####################################################")
    
    if not GEMINI_API_KEY:
        print("🛑 ERROR: GEMINI_API_KEY is not set. Cannot proceed with AI steps.")
        return

    # 1. Download the file from Google Drive (Step 7)
    print(f"-> [Step 7] Downloading MP4 from: {recording_url}")
    # TODO: Implement robust Google Drive download logic (Requires Google Drive API setup)
    
    # 2. Transcribe the audio (Step 7)
    print("-> [Step 7] Generating transcript...")
    # TODO: Implement Transcription (e.g., using google-genai or Google Cloud STT)
    
    # 3. Summarize (Offline) and Fine-tune (Step 8)
    print("-> [Step 8] Cleaning transcript and generating OFFLINE summary...")
    # TODO: Implement LLM call using the transcript and prompt engineering for format/style/length.
    
    # 4. Summarize (Online/RAG) and Augment with Graphics (Step 9)
    print("-> [Step 9] Generating final summary with web search and graphics...")
    # TODO: Implement LLM + Search (RAG) call for final, augmented summary.

    print("\nProcess Structure Complete! Implementation of AI steps is next.")

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
    asyncio.run(main())
    