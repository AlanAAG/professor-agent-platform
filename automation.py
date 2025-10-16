import os
import asyncio
import argparse
import re
import requests # New import for authenticated download
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from openai import OpenAI # For Step 7 (Transcription)
# from google import genai # For Steps 8 & 9 (Summarization)

# Load environment variables from the .env file
load_dotenv()

# --- Configuration ---
LOGIN_URL = "https://coach.tetr.com/"
COURSES_URL = LOGIN_URL + "courses"
AUTH_STATE_FILE = "auth_state.json"

# Retrieve credentials securely from environment variables
USERNAME = os.environ.get("COACH_USERNAME")
PASSWORD = os.environ.get("COACH_PASSWORD")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Ensure this is set in .env if used

# --- Playwright Selectors ---
MODAL_HEADER_SELECTOR = '.popupHeader' 
SVG_CLOSE_SELECTOR = f'{MODAL_HEADER_SELECTOR} svg' 

# --- Helper Function: Session Validation ---
async def is_session_valid(page):
    """Checks if the saved session can successfully load the dashboard."""
    try:
        await page.goto(LOGIN_URL) 
        await page.wait_for_selector('#gtm-IdDashboard', timeout=5000)
        return True
    except Exception:
        return False

# --- Utility Function: Extract Drive ID ---
def extract_drive_file_id(url: str) -> str:
    """Extracts the file ID from a Google Drive URL."""
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    raise ValueError("Invalid Google Drive URL format.")

# ----------------------------------------------------------------------
# --- CORE LOGIC: DOWNLOAD VIA COOKIES (Step 7 - Part 1) ---
# ----------------------------------------------------------------------
def download_drive_file_with_cookies(file_id: str, cookies: list, local_filename: str):
    """
    Constructs the download URL and fetches the file using cookies from the authenticated Playwright session.
    """
    # 1. Prepare the requests session
    session = requests.Session()
    
    # 2. Convert Playwright cookies (list of dicts) to requests format (dict)
    requests_cookies = {}
    for cookie in cookies:
        # We only care about cookies valid for the Google domain
        if 'google.com' in cookie['domain'] or '.google.com' in cookie['domain']:
            requests_cookies[cookie['name']] = cookie['value']

    session.cookies.update(requests_cookies)

    # 3. Construct the download URL
    # We use the direct download structure
    download_url = f"https://docs.google.com/uc?export=download&id={file_id}"
    
    print(f"-> Initiating authenticated download...")
    
    # Send the request with the authenticated session
    response = session.get(download_url, stream=True, allow_redirects=True)
    
    # Check for success (Google Drive sends a Content-Disposition header on success)
    if response.status_code == 200 and 'Content-Disposition' in response.headers:
        # Write file content in chunks
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ File successfully downloaded to {local_filename}")
        return local_filename
    
    else:
        # Handle cases where authentication failed or file permissions are wrong
        response.raise_for_status()
        raise Exception(f"Download failed: Check file sharing/authentication. Status: {response.status_code}")


# ----------------------------------------------------------------------
# --- Core Login and Navigation Functions (Omitted for brevity, kept structure) ---
# ----------------------------------------------------------------------

# NOTE: The perform_login and is_session_valid functions remain unchanged from the previous complete script.

# ----------------------------------------------------------------------
# --- AI PROCESSING LOGIC (Steps 7, 8, 9) ---
# ----------------------------------------------------------------------
async def process_recording_file(recording_url: str):
    """
    Handles file download, transcription (Whisper), and summarization (Gemini).
    """
    print("\n\n#####################################################")
    print("# STARTING AI PROCESS (Steps 7, 8, 9) #")
    print("#####################################################")
    
    # Check for necessary API keys
    if not OPENAI_API_KEY:
        print("🛑 ERROR: OPENAI_API_KEY is not set. Cannot proceed with Whisper transcription.")
        return
    if not GEMINI_API_KEY:
        print("🛑 ERROR: GEMINI_API_KEY is not set. Cannot proceed with Gemini summarization.")
        return

    # --- Setup ---
    # client_gemini = genai.Client(api_key=GEMINI_API_KEY) # Uncomment when using genai
    client_openai = OpenAI(api_key=OPENAI_API_KEY)
    file_id = extract_drive_file_id(recording_url)
    local_video_path = f"lecture_{file_id}.mp4"
    transcript = None
    
    # 1. Download the file (Step 7 - Part 1)
    print(f"-> [Step 7] Downloading MP4 from: {recording_url}")
    
    # NOTE: The download is handled inside search_and_download_lecture (prior step)
    
    # 2. Transcribe the audio (Step 7 - Part 2: OpenAI Whisper)
    print("-> [Step 7] Generating transcript using OpenAI Whisper...")
    
    try:
        # IMPORTANT: Whisper API supports MP4 directly, but file size limit is 25MB.
        # For larger files, you would need to use a dedicated audio extraction library (e.g., moviepy).
        with open(local_video_path, "rb") as audio_file:
            transcript_response = client_openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        transcript = transcript_response
        print("✅ Transcription complete.")
        
    except FileNotFoundError:
        print(f"❌ Transcription failed: Local file not found at {local_video_path}. Check download step.")
        return 
    except Exception as e:
        print(f"❌ OpenAI Whisper API call failed: {e}")
        return
        
    finally:
        # Clean up the temporary file (CRITICAL for Codespace management)
        if os.path.exists(local_video_path):
            os.remove(local_video_path)
            print(f"🧹 Cleaned up temporary file: {local_video_path}")


    # 3. Summarize (Offline) and Fine-tune (Step 8)
    print("-> [Step 8] Cleaning transcript and generating OFFLINE summary...")
    # TODO: Implement LLM call using the transcript and prompt engineering for format/style/length.
    # offline_summary = client_gemini.models.generate_content(...) 
    
    # 4. Summarize (Online/RAG) and Augment with Graphics (Step 9)
    print("-> [Step 9] Generating final summary with web search and graphics...")
    # TODO: Implement LLM + Search (RAG) call for final, augmented summary.
    
    print("\nProcess Structure Complete! Time to implement Steps 8 & 9.")


# ----------------------------------------------------------------------
# --- Core Automation Logic (Steps 3, 4, 5, 6) ---
# ----------------------------------------------------------------------
async def search_and_download_lecture(page, context, course_name: str, lecture_title: str, lecture_date: str):
    """
    Handles navigation, subject/lecture selection, and executes the download.
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

    # --- Step 5c/6: Locate Session, Open Modal, Click Download Link ---
    print(f"\n[Step 5c] -> Locating session item for date: {lecture_date}")
    
    # Selectors
    VIEW_ALL_SELECTOR = 'div.view:has-text("View All")'
    MODAL_HEADER_SELECTOR = '.popupHeader' 
    RECORDING_LINK_SELECTOR = 'a:has-text("Click to View")'
    SVG_CLOSE_SELECTOR = f'{MODAL_HEADER_SELECTOR} svg' 
    
    recording_url = None
    try:
        # 1. Wait for session list to render, find item, and click to expand
        await page.wait_for_selector('li:has-text("PM")', timeout=15000) 
        date_element_xpath = f"//p[@class='date' and contains(., '{lecture_date}')]"
        session_list_item = page.locator(f'{date_element_xpath}/ancestor::li').first
        await session_list_item.click(force=True, timeout=10000) 
        
        # 2. Click "View All" to open modal
        view_all_locator = page.locator(VIEW_ALL_SELECTOR)
        await view_all_locator.wait_for(state="visible", timeout=10000)
        await view_all_locator.click(force=True, timeout=15000) 
        
        # 3. Wait for modal to load, and extract the URL
        await page.wait_for_selector(MODAL_HEADER_SELECTOR, timeout=30000) 
        await page.wait_for_selector('h5:has-text("Session Recording")', timeout=10000) 
        
        recording_link_locator = page.get_by_text("Click to View", exact=True).first
        await recording_link_locator.wait_for(state="visible", timeout=10000) 

        # --- CRITICAL STEP: Download Prep ---
        # Get all cookies from the context *before* the new tab opens
        google_drive_cookies = await context.cookies(urls=[recording_url])
        file_id = extract_drive_file_id(recording_url)
        local_video_path = f"lecture_{file_id}.mp4"

        # 4. Execute Click and Get URL
        async with context.expect_page() as new_page_info:
            await recording_link_locator.click(force=True)
        
        recording_page = await new_page_info.value
        await recording_page.wait_for_load_state("load")
        recording_url = recording_page.url
        print(f"✅ Recording link clicked! New URL (Google Drive): {recording_url}")

        # 5. Execute Download using the extracted cookies
        download_drive_file_with_cookies(file_id, google_drive_cookies, local_video_path)
        
        # 6. Cleanup
        await page.click(SVG_CLOSE_SELECTOR) 
        await recording_page.close()
        
    except Exception as e:
        print(f"❌ Error during navigation or download: {e}")
        await page.screenshot(path="final_download_error.png")
        return
        
    # --- PROCEED TO AI STEPS (7, 8, 9) ---
    await process_recording_file(local_video_path)


# ----------------------------------------------------------------------
# --- Main Execution (Handles User Arguments) ---
# ----------------------------------------------------------------------
# NOTE: The main execution block remains unchanged.
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
        # NOTE: Pass the context for cookie extraction
        await search_and_download_lecture(page, context, TARGET_COURSE, TARGET_LECTURE_TITLE, TARGET_LECTURE_DATE)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
    