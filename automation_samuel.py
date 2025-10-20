# ==============================================================================
#  Automated Course Transcription Agent for Google Colab
#  FINAL STABLE VERSION - Corrected Selectors & Logic
# ==============================================================================
#
# INSTRUCTIONS:
# 1. Open a new notebook in Google Colab (https://colab.research.google.com/).
# 2. Copy and paste all the code from this file into a single cell in your Colab notebook.
# 3. Your username and password have been saved in the script.
# 4. Run the cell.
#
# ==============================================================================

#@title Step 1: Install Necessary Libraries
!pip install -q --upgrade selenium google-api-python-client google-auth-httplib2 google-auth-oauthlib beautifulsoup4

#@title Step 2: Import Libraries and Set Up Google Authentication
import os
import re
import time
from getpass import getpass
from datetime import datetime

# Selenium imports for web automation
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException, ElementClickInterceptedException, NoSuchFrameException

# Google API imports
from google.colab import files
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# HTML Parsing
from bs4 import BeautifulSoup

# --- Google Authentication Setup ---
SCOPES = ["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"]
creds = None
if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        print("Please upload your client secret JSON file.")
        uploaded = files.upload()
        json_files = [filename for filename in uploaded.keys() if filename.endswith('.json')]
        if not json_files: raise ValueError("Upload failed. No JSON credentials file was found.")
        credentials_filename = json_files[0]
        print(f"Found credentials file: '{credentials_filename}'. Proceeding with authentication.")

        flow = InstalledAppFlow.from_client_secrets_file(
            credentials_filename,
            SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )

        auth_url, _ = flow.authorization_url(prompt='consent')
        print('Please visit this URL to authorize this application:')
        print(auth_url)
        code = input('Enter the authorization code here: ')
        flow.fetch_token(code=code)
        creds = flow.credentials

    with open("token.json", "w") as token:
        token.write(creds.to_json())

try:
    docs_service = build("docs", "v1", credentials=creds)
    print("Successfully authenticated and connected to Google Docs API.")
except Exception as e:
    print(f"An error occurred building the Docs service: {e}")
    raise

#@title Step 3: Helper Functions

def clean_transcription(raw_text):
    no_timestamps = re.sub(r'\[?\d{1,2}:\d{1,2}(:\d{1,2})?\]?\s*', '', raw_text)
    paragraph = no_timestamps.replace('\n', ' ').strip()
    cleaned_paragraph = re.sub(r'\s+', ' ', paragraph)
    return cleaned_paragraph

def append_to_google_doc(document_id, text_to_append):
    if not text_to_append:
        print(f"Skipping document {document_id} because there is no text to append.")
        return
    try:
        document = docs_service.documents().get(documentId=document_id).execute()
        end_index = document.get('body').get('content')[-1].get('endIndex') - 1
        requests = [{'insertText': {'location': {'index': end_index},'text': "\n\n" + text_to_append}}]
        docs_service.documents().batchUpdate(documentId=document_id, body={'requests': requests}).execute()
        print(f"Successfully appended transcription to Google Doc ID: {document_id}")
    except HttpError as err:
        print(f"An error occurred while writing to Google Doc ({document_id}): {err}")

def parse_session_date(date_str):
    date_part = date_str.split('|')[0].strip()
    try:
        return datetime.strptime(date_part, '%d/%m/%Y')
    except ValueError:
        print(f"Warning: Could not parse date string '{date_str}'")
        return None

#@title Step 4: The Main Automation Script (Refactored and Finalized)
def run_transcription_agent():
    """
    The main function that logs into the site, scrapes, and updates Google Docs.
    This version uses the "Resources" tab and a highly stable navigation loop.
    """

    # --- CONFIGURATION ---

    COACH_LOGIN_URL = "https://coach.tetr.com/login"
    COACH_MAIN_URL = "https://coach.tetr.com/"
    COURSES_PAGE_URL = "https://coach.tetr.com/courses"

    # --- AUTOMATED CREDENTIALS ---
    USERNAME = "samuel.estrada_bmt2029@tetr.com"
    PASSWORD = "Csem71425$"

    CUTOFF_DATE = datetime(2025, 10, 9)

    SUBJECT_TO_DOC_ID = {
        "AIML101 How do machines see, hear or speak": "1NudFRWPDnFjbQxm8SpzNZZLqS7XkjTrRWOobZVWcZQI",
        "PRTC301 How to use statistics to build a better business": "1N6KPFn0YyIN8JZt_kINs08TrqnipA1K9WvSepmuqpNQ",
        "PRTC201 How to get comfortable with excel": "1zsy8m4zxuQxspJ-d88neu0_xTtRzJe_MemVt54B-N60",
        "FIFI101 How to understand basic financial terminology": "1KnBR8xsH_EPbkGDqv05UHxjNywOpaqWt2FnZ3YaRSWo",
        "LA101 How to decode global trends and navigate economic transformations": "1kIdRF8ChqQ_0Z9WFdN-sIm9dz0z8g2tPZzUguTO0flM",
        "MAST102 How to read market for better decision making": "1nMpNvF33l7j45YSiGTHkuy5eAJlXtD1Xo87vzqte_3M",
        "SAMA101 How to identify gaps in the market": "1g7O1AzCm1_Q7tuPTfLZ2tMuctJ9ZzMek3hl_c7L4uvs",
        "SAMA401 How to execute digital marketing on Meta": "1lv9wFqrI5goQ4HVNdjv7TdGANt4JQqXe_WZvVyK7AhM",
        "SAMA502 How to execute CRO and increase AOV": "1ZdEZ86OqWZ0J_a1mlxHCb_31kCNvQlh3ooFbuEqZ1hA",
        "MAST401 How to validate, shape, and launch a startup": "10IRAYyUreW5wfs9Klm3qObo-8D434TQE4s2ContQQkk"
    }

    # ========================= COURSE GROUP MAPPING =========================
    COURSE_TO_GROUP = {
        "AIML101": "Quantitative Tools for Business",
        "PRTC301": "Quantitative Tools for Business",
        "PRTC201": "Quantitative Tools for Business",
        "MAST401": "Management Project - I",
        "FIFI101": "Management Accounting",
        "LA101":   "Microeconomics",
        "MAST102": "Microeconomics",
        "SAMA101": "Marketing Strategies",
        "SAMA401": "Marketing Strategies",
        "SAMA502": "Marketing Strategies"
    }
    # List of courses that are visible by default and don't require group expansion.
    DEFAULT_VISIBLE_COURSES = {"AIML101", "PRTC301", "PRTC201"}
    # ===========================================================================

    # --- SELECTORS (FINALIZED FOR 'RESOURCES' FLOW) ---
    RESOURCES_TAB_SELECTOR = (By.XPATH, "//div[contains(@class, 'sc-Rbkqr')]//h4[contains(text(), 'Resources')]")
    SESSIONS_RECORDINGS_LINK_SELECTOR = (By.XPATH, "//div[contains(@class, 'sc-kRJjUj')]//p[contains(text(), 'Session Recordings')]")
    RECORDING_ITEM_SELECTOR = (By.CLASS_NAME, "fileBox")

    ZOOM_TRANSCRIPTION_SELECTOR = (By.CLASS_NAME, "transcript-container")
    # This selector finds the main container for the transcript.
    DRIVE_TRANSCRIPTION_SELECTOR = (By.CSS_SELECTOR, "div[jsname='h7hTqc']")

    # --- End of Configuration ---

    options = webdriver.ChromeOptions()
    options.add_argument("--headless"); options.add_argument("--no-sandbox"); options.add_argument("--disable-dev-shm-usage")

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 60)

        print("Navigating to login page..."); driver.get(COACH_LOGIN_URL)
        print("Entering credentials...")
        wait.until(EC.presence_of_element_located((By.NAME, "officialEmail"))).send_keys(USERNAME)
        driver.find_element(By.NAME, "password").send_keys(PASSWORD)
        driver.find_element(By.ID, "gtmLoginStd").click()

        print("Login submitted. Waiting for navigation...")
        try:
            wait.until(EC.url_to_be(COACH_MAIN_URL))
            print("Successfully reached the main page after login.")
        except TimeoutException:
            print("Login failed. Could not navigate to the main page. Check credentials or website changes.")
            raise

        for subject_name, doc_id in SUBJECT_TO_DOC_ID.items():
            print(f"\n--- Processing Subject: {subject_name} ---")
            try:
                driver.get(COURSES_PAGE_URL)
                course_code = subject_name.split(' ')[0]

                # ========================= FINAL ROBUST NAVIGATION LOGIC =========================
                try:
                    if course_code in DEFAULT_VISIBLE_COURSES:
                        # --- PATH 1: For courses visible by default ---
                        print(f"Course {course_code} is a special case. Searching directly...")
                        target_course_link_xpath = f"//a[contains(@href, 'courseCode={course_code}')]"
                        subject_link_element = wait.until(EC.presence_of_element_located((By.XPATH, target_course_link_xpath)))
                        print(f"Clicking course link '{course_code}'...")
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", subject_link_element)
                        time.sleep(1)
                        driver.execute_script("arguments[0].click();", subject_link_element)
                    else:
                        # --- PATH 2: For all other courses that require group expansion ---
                        group_name = COURSE_TO_GROUP.get(course_code)
                        if not group_name:
                            print(f"Warning: Group for course {course_code} not defined. Skipping.")
                            continue

                        print(f"Waiting for group header '{group_name}' to be visible...")
                        group_header_xpath = f"//div[contains(@class, 'domainHeader')]//p[contains(@class, 'title') and normalize-space()='{group_name}']"
                        target_group_element = wait.until(EC.visibility_of_element_located((By.XPATH, group_header_xpath)))

                        print(f"Attempting to expand group '{group_name}'...")
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_group_element)
                        time.sleep(1)
                        target_group_element.click()
                        print(f"Successfully clicked group '{group_name}'.")

                        print("Pausing for 5 seconds to allow content to begin loading...")
                        time.sleep(5)

                        target_course_link_xpath = f"//a[contains(@href, 'courseCode={course_code}')]"
                        subject_link_element = wait.until(EC.presence_of_element_located((By.XPATH, target_course_link_xpath)))

                        print(f"Clicking course link '{course_code}'...")
                        driver.execute_script("arguments[0].click();", subject_link_element)

                except TimeoutException:
                    print(f"Info: Timed out waiting for '{subject_name}'. The course may not be available or page structure changed.")
                    continue
                # =====================================================================

                print(f"Navigated to '{subject_name}' page.")
                course_page_url = driver.current_url

                print("Looking for 'Resources' tab...");
                resources_tab = wait.until(EC.element_to_be_clickable(RESOURCES_TAB_SELECTOR))
                driver.execute_script("arguments[0].click();", resources_tab)

                short_wait = WebDriverWait(driver, 5)
                try:
                    print("Looking for 'Session Recordings' link (max 5s wait)...");
                    session_recordings_link = short_wait.until(EC.element_to_be_clickable(SESSIONS_RECORDINGS_LINK_SELECTOR))
                    driver.execute_script("arguments[0].click();", session_recordings_link)
                except TimeoutException:
                    print(f"Info: 'Session Recordings' link not found for '{subject_name}'. Skipping subject.")
                    continue

                print("Waiting for initial recordings list to load...")
                initial_recordings_list = wait.until(EC.presence_of_all_elements_located(RECORDING_ITEM_SELECTOR))
                num_recordings = len(initial_recordings_list)
                print(f"Found {num_recordings} total recordings.")

                if not num_recordings > 0:
                    print("No recordings found. Skipping subject.")
                    continue

                new_recordings_found = False
                for index in range(num_recordings):
                    try:
                        # ... Re-navigation logic ...
                        print(f"\n--- Re-navigating to recordings list for item {index + 1}/{num_recordings} ---")
                        driver.get(course_page_url)

                        print("Looking for 'Resources' tab...");
                        resources_tab = wait.until(EC.element_to_be_clickable(RESOURCES_TAB_SELECTOR))
                        driver.execute_script("arguments[0].click();", resources_tab)

                        print("Looking for 'Session Recordings' link...");
                        session_recordings_link = wait.until(EC.element_to_be_clickable(SESSIONS_RECORDINGS_LINK_SELECTOR))
                        driver.execute_script("arguments[0].click();", session_recordings_link)

                        print("Waiting for recordings list to be present...")
                        all_recordings = wait.until(EC.presence_of_all_elements_located(RECORDING_ITEM_SELECTOR))
                        current_recording_item = all_recordings[index]

                        date_element = current_recording_item.find_element(By.TAG_NAME, "span")
                        date_text = date_element.get_attribute("textContent").strip()
                        if not date_text:
                            print("Found a recording item without a date. Skipping.")
                            continue
                        recording_date = parse_session_date(date_text)

                        if recording_date is None or recording_date < CUTOFF_DATE:
                            print(f"Recording dated '{date_text}' is old or unparsable. Skipping.")
                            continue

                        new_recordings_found = True
                        print(f"Found new recording dated: {recording_date.strftime('%Y-%m-%d')}. Processing...")

                        link_to_click = current_recording_item.find_element(By.XPATH, ".//ancestor::a[1]")
                        original_window = driver.current_window_handle
                        driver.execute_script("arguments[0].click();", link_to_click)
                        print("Clicked recording link.")

                        wait.until(EC.number_of_windows_to_be(2))
                        new_window_handle = [wh for wh in driver.window_handles if wh != original_window][0]
                        driver.switch_to.window(new_window_handle)

                        time.sleep(4)
                        current_url = driver.current_url

                        raw_transcription = ""
                        if "drive.google.com" in current_url:
                            print("Google Drive page detected. Applying final double-wait extraction logic...")
                            try:
                                # STAGE 1: Wait for the main container to be present.
                                print("Waiting for the main transcript container...")
                                main_container = wait.until(EC.presence_of_element_located(DRIVE_TRANSCRIPTION_SELECTOR))

                                # STAGE 2: CRUCIAL - Wait for the *first text segment* to appear within the container.
                                print("Now, waiting for the first text segment to render inside the container...")
                                first_segment_selector = (By.CSS_SELECTOR, "div[jsname='h7hTqc'] div.wyBDIb")
                                wait.until(EC.presence_of_element_located(first_segment_selector))

                                # If both waits succeed, the elements are present. Now we can extract them all.
                                print("Success! Segments are present. Extracting all text...")
                                text_segment_elements = driver.find_elements(By.CSS_SELECTOR, "div.wyBDIb")
                                all_text_fragments = [elem.get_attribute("textContent") for elem in text_segment_elements]
                                raw_transcription = " ".join(filter(None, all_text_fragments))

                            except TimeoutException:
                                print("Warning: Timed out waiting for the transcript to fully render. The page structure may have changed or is taking too long to load.")
                            except Exception as e:
                                print(f"An error occurred during Google Drive extraction: {e}")

                        elif "zoom.us" in current_url:
                            print("Zoom page detected.");
                            transcription_element = wait.until(EC.presence_of_element_located(ZOOM_TRANSCRIPTION_SELECTOR))
                            raw_transcription = transcription_element.get_attribute("textContent")
                        else:
                            print(f"Warning: Unknown recording platform at {current_url}. Skipping.")
                            driver.close(); driver.switch_to.window(original_window)
                            continue

                        # This message now accurately reflects the state of raw_transcription
                        if raw_transcription:
                            print("Found transcription text.")
                            cleaned_paragraph = clean_transcription(raw_transcription)
                            append_to_google_doc(doc_id, cleaned_paragraph)
                        else:
                            print("No transcription text was extracted.")


                        print("Closing transcription window..."); driver.close(); driver.switch_to.window(original_window)

                    except (NoSuchElementException, TimeoutException, StaleElementReferenceException) as e:
                        print(f"Skipping a recording due to an error. Error: {e}")
                        continue

                if not new_recordings_found:
                    print("No new recordings found for this subject after the cutoff date.")

            except Exception as e:
                print(f"An error occurred processing '{subject_name}': {e}")
                continue

    finally:
        if driver:
            driver.quit()
        print("\nAgent finished its run. Browser closed.")

#@title Step 5: Run the Agent
if __name__ == "__main__":
    run_transcription_agent()