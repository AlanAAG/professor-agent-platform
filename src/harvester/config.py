# src/harvester/config.py

# --- Core URLs ---
LOGIN_URL = "https://coach.tetr.com/login"
BASE_URL = "https://coach.tetr.com/"
COURSES_URL = BASE_URL + "courses"
AUTH_STATE_FILE = "data/auth_state.json" # Relative to project root

"""
Selectors and course mappings adapted to match the partner Selenium script,
converted to Playwright-friendly CSS/XPath where appropriate.
"""

# --- Navigation Selectors ---
# Login Page
USERNAME_SELECTOR = 'input[name="officialEmail"]'
PASSWORD_SELECTOR = 'input[name="password"]'
LOGIN_BUTTON_SELECTOR = '#gtmLoginStd'
DASHBOARD_INDICATOR = '#gtm-IdDashboard'  # Selector to confirm login success

# Courses Page
# Group header: partner script uses an XPath searching for div.domainHeader with a nested p.title == group_name
# Playwright equivalent using :has and :has-text
GROUP_HEADER_SELECTOR = "div.domainHeader:has(p.title:has-text('{group_name}'))"

# Course link: same logic as partner (href contains courseCode=<code>)
COURSE_LINK_SELECTOR = "a[href*='courseCode={course_code}']"

# Course Details Page (Resources Tab Navigation)
# Partner script targets a container then an h4 with 'Resources'. Keep a robust fallback to bare h4.
RESOURCES_TAB_SELECTOR = "h4:text-is('Resources')"

# Resource Section Headers (ensure div.dlvLeftHeader:has-text(...) pattern)
RECORDINGS_LINK_SELECTOR = "div.dlvLeftHeader:has-text('Session Recordings')"
PRE_READ_SECTION_SELECTOR = "div.dlvLeftHeader:has-text('Pre-Read Materials')"
IN_CLASS_SECTION_SELECTOR = "div.dlvLeftHeader:has-text('In Class Materials')"
POST_CLASS_SECTION_SELECTOR = "div.dlvLeftHeader:has-text('Post Class Materials')"

# Resource items and sub-elements
RECORDING_ITEM_SELECTOR = "div.fileBox"  # Container for each recording link/info
RESOURCE_ITEM_SELECTOR = "div.fileBox"
RESOURCE_TITLE_SELECTOR = "div.fileContentCol p"
RESOURCE_DATE_SELECTOR = "div.fileContentCol span"
RESOURCE_LINK_SELECTOR = "div.fileBox > a"

# (Removed) Static syllabus selector; syllabus scraping no longer used

# --- Transcript Scraping Selectors (kept; used by scraping module) ---
# Google Drive Web Viewer
DRIVE_VIDEO_PLAY_BUTTON = 'button[jsname="dW8tsb"]'
DRIVE_TRANSCRIPT_SEGMENT_SELECTOR = "div[jsname='h7hTqc'] div.wyBDIb"
DRIVE_SETTINGS_BUTTON = 'button[jsname="J7HKb"]'
DRIVE_TRANSCRIPT_MENU_ITEM = 'div[role^="menuitem"]:has-text("Transcript")'
# Zoom Web Viewer
ZOOM_TRANSCRIPT_CONTAINER_SELECTOR = "div.transcript-container"

# --- Course Mappings ---
# 1) Existing map from this project (renamed to LEGACY for merge step)
LEGACY_COURSE_MAP = {
    # Quantitative Tools for Business
    "AIML101": {"name": "AIML", "group": "Quantitative Tools for Business"},
    "PRTC301": {"name": "Statistics", "group": "Quantitative Tools for Business"},
    "PRTC201": {"name": "Excel", "group": "Quantitative Tools for Business"},
    # Mathematics for Engineers
    "CAL101": {"name": "Calculus", "group": "Mathematics for Engineers"},
    # Management Project - I
    "MAST401": {"name": "Startup", "group": "Management Project - I"},
    "CAP101": {"name": "Dropshipping", "group": "Management Project - I"},
    "COMM101": {"name": "PublicSpeaking", "group": "Management Project - I"},
    "MAST601": {"name": "Networking", "group": "Management Project - I"},
    # Computer Science
    "CS101": {"name": "OOP", "group": "Computer Science"},
    # Management Accounting
    "FIFI101": {"name": "FinanceBasics", "group": "Management Accounting"},
    "MAST102": {"name": "MarketAnalysis", "group": "Management Accounting"},
    # Marketing Strategies
    "SAMA101": {"name": "MarketGaps", "group": "Marketing Strategies"},
    "SAMA401": {"name": "MetaMarketing", "group": "Marketing Strategies"},
    "SAMA502": {"name": "CRO", "group": "Marketing Strategies"},
}

# 2) Partner course subjects (from SUBJECT_TO_DOC_ID keys)
_PARTNER_SUBJECTS = [
    "AIML101 How do machines see, hear or speak",
    "PRTC301 How to use statistics to build a better business",
    "PRTC201 How to get comfortable with excel",
    "FIFI101 How to understand basic financial terminology",
    "LA101 How to decode global trends and navigate economic transformations",
    "MAST102 How to read market for better decision making",
    "SAMA101 How to identify gaps in the market",
    "SAMA401 How to execute digital marketing on Meta",
    "SAMA502 How to execute CRO and increase AOV",
    "MAST401 How to validate, shape, and launch a startup",
    "COMM101 How to own a stage",
    "DRP101 How to build a dropshipping business",
    "MAST601 How to network effortlessly",
]

# 3) Partner group mapping (from COURSE_TO_GROUP)
PARTNER_COURSE_TO_GROUP = {
    "AIML101": "Quantitative Tools for Business",
    "PRTC301": "Quantitative Tools for Business",
    "PRTC201": "Quantitative Tools for Business",
    "MAST401": "Management Project - I",
    "FIFI101": "Management Accounting",
    "LA101": "Microeconomics",
    "MAST102": "Microeconomics",
    "SAMA101": "Marketing Strategies",
    "SAMA401": "Marketing Strategies",
    "SAMA502": "Marketing Strategies",
    "COMM101": "Management Project - I",
    "DRP101": "Management Project - I",
    "MAST601": "Management Project - I",
}

# 4) Build partner-derived map: code -> { full_name, code, group }
_partner_map: dict[str, dict] = {}
for subj in _PARTNER_SUBJECTS:
    parts = subj.split(" ", 1)
    if not parts:
        continue
    code = parts[0]
    friendly_name = parts[1].strip() if len(parts) > 1 else code
    _partner_map[code] = {
        "full_name": friendly_name,
        "code": code,
        "group": PARTNER_COURSE_TO_GROUP.get(code),
    }

# 5) Merge legacy with partner
#    CRITICAL: 'name' MUST remain the short alias used throughout the app
#    (and as keys in persona.json) so Supabase filtering matches correctly.
COURSE_MAP: dict[str, dict] = {}
all_codes = set(LEGACY_COURSE_MAP.keys()) | set(_partner_map.keys())
for code in sorted(all_codes):
    partner_entry = _partner_map.get(code)
    legacy_entry = LEGACY_COURSE_MAP.get(code)
    merged = {
        # Keep the canonical short name (e.g., "AIML", "Excel", "Statistics")
        "name": (legacy_entry or {}).get("name") or code,
        "code": code,
        # Prefer partner group when available
        "group": (partner_entry or {}).get("group") if partner_entry and partner_entry.get("group") is not None else (legacy_entry or {}).get("group"),
        # Preserve the human-friendly title for display/logging when available
        "full_name": (partner_entry or {}).get("full_name") or (legacy_entry or {}).get("name") or code,
    }
    COURSE_MAP[code] = merged

# 6) Default visible courses (from partner script)
DEFAULT_VISIBLE_COURSES = {"AIML101", "PRTC301", "PRTC201"}

# --- Other Settings ---
# Cutoff date logic handled dynamically in the pipeline script (mode='daily'/'backlog')
# Output directories defined in pipeline script