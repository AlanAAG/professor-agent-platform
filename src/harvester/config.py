# src/harvester/config.py

# --- Core URLs ---
LOGIN_URL = "https://coach.tetr.com/login"
BASE_URL = "https://coach.tetr.com/"
COURSES_URL = BASE_URL + "courses"
AUTH_STATE_FILE = "data/auth_state.json" # Relative to project root

# --- Navigation Selectors ---
# Login Page
USERNAME_SELECTOR = 'input[name="officialEmail"]'
PASSWORD_SELECTOR = 'input[name="password"]'
LOGIN_BUTTON_SELECTOR = '#gtmLoginStd'
DASHBOARD_INDICATOR = '#gtm-IdDashboard' # Selector to confirm login success

# Courses Page
# Group header (XPath based on previous Selenium script)
GROUP_HEADER_XPATH = "//div[contains(@class, 'domainHeader')]//p[contains(@class, 'title') and normalize-space()='{group_name}']"
# Course link (XPath based on previous Selenium script)
COURSE_LINK_SELECTOR = "a[href*='courseCode={course_code}']"

# Course Details Page (Resources Tab Navigation)
# Using more specific selectors if possible is better, these are based on text/structure
RESOURCES_TAB_SELECTOR = "button:has-text('Resources'), h4:has-text('Resources')" # Broad selector for tab
RECORDINGS_LINK_SELECTOR = "p:has-text('Session Recordings')" # Specific text within resources
RECORDING_ITEM_SELECTOR = "div.fileBox" # Container for each recording link/info (adjust if needed)
# Add selectors for resource items (PDFs, links) within Resources tab if different from recordings
RESOURCE_ITEM_SELECTOR = "div.fileBox" # Example, might be different class
RESOURCE_TITLE_SELECTOR = "div.fileContentCol p"
RESOURCE_DATE_SELECTOR = "div.fileContentCol span"
RESOURCE_LINK_SELECTOR = "div.fileBox > a"

# Static Content Selectors (Example - Needs refinement by inspecting page)
SYLLABUS_CONTAINER_SELECTOR = "div.course-description-container" # Placeholder

# --- Transcript Scraping Selectors ---
# Google Drive Web Viewer
DRIVE_VIDEO_PLAY_BUTTON = 'button[jsname="dW8tsb"]'
DRIVE_TRANSCRIPT_SEGMENT_SELECTOR = "div[jsname='h7hTqc'] div.wyBDIb"
DRIVE_SETTINGS_BUTTON = 'button[jsname="J7HKb"]'
DRIVE_TRANSCRIPT_MENU_ITEM =  'div[role^="menuitem"]:has-text("Transcript")'
# Zoom Web Viewer
ZOOM_TRANSCRIPT_CONTAINER_SELECTOR = "div.transcript-container" # Adjust if needed

# --- Course Mappings ---
# Mapping course codes to their display name and group (if needed for expansion)
# None for group means it's likely visible by default or we don't need group expansion for it
COURSE_MAP = {
    # Quantitative Tools for Business
    "AIML101": {"name": "How do machines see, hear or speak", "group": "Quantitative Tools for Business"},
    "PRTC301": {"name": "How to use statistics to build a better business", "group": "Quantitative Tools for Business"},
    "PRTC201": {"name": "How to get comfortable with excel", "group": "Quantitative Tools for Business"},
    #Mathematics for Engineers
    "CAL101": {"name": "How to use calculus for real world problem solving", "group": "Mathematics for Engineers"},
    # Management Project -I
    "MAST401": {"name": "How to validate, shape, and launch a startup", "group": "Management Project - I"},
    "CAP101": {"name": "How to build a Dropshipping business", "group": "Management Project - I"}, # Added from video
    "COMM101": {"name": "How to own a stage", "group": "Management Project - I"},
    "MAST601": {"name": "How to network effortlessly","group": "Management Project - I"},
    #Computer Science
    "CS101": {"name": "How do software developers use OOP to design intelligent systems for startups", "group": "Computer Science"},
    # Management Accounting
    "FIFI101": {"name": "How to understand basic financial terminology", "group": "Management Accounting"},
    "MAST102": {"name": "How to read market for better decision making", "group": "Management Accounting"},
    # Marketing Strategies
    "SAMA101": {"name": "How to identify gaps in the market", "group": "Marketing Strategies"},
    "SAMA401": {"name": "How to execute digital marketing on Meta", "group": "Marketing Strategies"},
    "SAMA502": {"name": "How to execute CRO and increase AOV", "group": "Marketing Strategies"},
    # Add other courses as needed
}

# Optional: List of courses that are known to be visible without expanding a group
# This can simplify navigation logic slightly if reliable
DEFAULT_VISIBLE_COURSES = {"AIML101", "PRTC301", "PRTC201"} # Example, verify

# --- Other Settings ---
# Cutoff date logic handled dynamically in the pipeline script (mode='daily'/'backlog')
# Output directories defined in pipeline script