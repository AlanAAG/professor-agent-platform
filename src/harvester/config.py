"""
Selenium-oriented configuration for the harvester module.

This converts prior Playwright selectors into robust Selenium-friendly
XPaths/CSS selectors and keeps course mappings and URLs separated from logic.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
import os

# --- Core URLs ---
LOGIN_URL = "https://coach.tetr.com/login"
BASE_URL = "https://coach.tetr.com/"
COURSES_URL = BASE_URL + "courses"

# File path to persist cookies/session if desired (used by Selenium helpers)
AUTH_STATE_FILE = "data/auth_state.json"

# --- Login Page Selectors (Selenium) ---
# Use simple CSS or IDs where stable; fallback to names
USERNAME_BY = ("css", 'input[name="officialEmail"]')
PASSWORD_BY = ("css", 'input[name="password"]')
LOGIN_BUTTON_BY = ("css", '#gtmLoginStd')

# A generic indicator that we are not on the login page anymore
DASHBOARD_INDICATOR_CSS = '#gtm-IdDashboard'

# --- Courses Page (Selenium) ---

# MODIFIED: More robust selector for group headers (avoid trailing space class names)
GROUP_HEADER_XPATH_TEMPLATE = (
    "//div[contains(@class,'domainHeader')][.//p[contains(@class,'title') and normalize-space(text())='{group_name}']]"
)

# Course link: anchor whose href contains courseCode={course_code}
# Trimmed stray space and kept partial href match for stability.
COURSE_LINK_XPATH_TEMPLATE = "//a[contains(@href, 'courseCode={course_code}')]"

# --- Course Details Page (Resources Tab Navigation) ---

# MODIFIED: Broadened selector to match both header tab and side nav item
# Prefer anchor/button elements that contain the text 'Resources', but keep the
# earlier header-based selector as a fallback to support older DOMs.
RESOURCES_TAB_XPATH = (
    "//a[contains(normalize-space(.), 'Resources')]"
    " | //button[contains(normalize-space(.), 'Resources')]"
    " | //div[contains(@class, 'sc-Rbkqr')]//h4[contains(normalize-space(.), 'Resources')]"
)

# MODIFIED: Selector from working Colab script
# Simpler XPath that finds the p tag by text inside the correct div container
SECTION_HEADER_XPATH_TPL = (
    "//div[contains(@class, 'sc-kRJjUj')]//p[contains(text(), '{section_title}')]"
)

PRE_READ_SECTION_TITLE = "Pre-Read Materials"
IN_CLASS_SECTION_TITLE = "In Class Materials"
POST_CLASS_SECTION_TITLE = "Post Class Materials"
SESSION_RECORDINGS_SECTION_TITLE = "Session Recordings"

# Resource items and sub-elements
# This selector ('fileBox') was correct and matches the Colab script.
RESOURCE_ITEM_CSS = "div.fileBox"
RESOURCE_TITLE_CSS = "div.fileContentCol p"
RESOURCE_DATE_CSS = "div.fileContentCol span"

# --- Transcript Scraping Selectors ---
# Google Drive Web Viewer
DRIVE_PLAY_BUTTON_CSS = "button[jsname='IGlMSc'], button[jsname='dW8tsb']"
DRIVE_SETTINGS_BUTTON_CSS = "button[jsname='dq27Te'], button[jsname='J7HKb']"
DRIVE_TRANSCRIPT_HEADING_CSS = "h2#ucc-0"
DRIVE_TRANSCRIPT_CONTAINER_CSS = "div[jsname='h7hTqc']"
DRIVE_TRANSCRIPT_SEGMENT_CSS = "div.JnEIz div.wyBDIb, div[jsname='h7hTqc'] div.wyBDIb"

# Zoom Web Viewer
ZOOM_TRANSCRIPT_CONTAINER_CSS = "div.transcript-container"
ZOOM_TRANSCRIPT_LIST_CSS = "ul.transcript-list"
ZOOM_TRANSCRIPT_TEXT_CSS = "div.timeline div.text"

# Initial interaction targets (best-effort)
ZOOM_INITIAL_INTERACTIONS = [
    "video",
    "div.player-container",
    "div#player",
    "div.playback-video",
    "canvas",
]

# --- Course Mappings ---
LEGACY_COURSE_MAP = {
    # Quantitative Tools for Business
    "AIML101": {"name": "AIML", "group": "Quantitative Tools for Business"},
    "PRTC301": {"name": "Statistics", "group": "Quantitative Tools for Business"},
    
    # --- MODIFIED: Moved PRTC201 ---
    # "PRTC201": {"name": "Excel", "group": "Quantitative Tools for Business"},
    
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
    # Corrected: PRTC201 is now in Management Accounting
    "PRTC201": {"name": "Excel", "group": "Management Accounting"},
    # Marketing Strategies
    "SAMA101": {"name": "MarketGaps", "group": "Marketing Strategies"},
    "SAMA401": {"name": "MetaMarketing", "group": "Marketing Strategies"},
    "SAMA502": {"name": "CRO", "group": "Marketing Strategies"},
}

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

PARTNER_COURSE_TO_GROUP = {
    "AIML101": "Quantitative Tools for Business",
    "PRTC301": "Quantitative Tools for Business",
    
    # Corrected: PRTC201 is in Management Accounting
    "PRTC201": "Management Accounting", 
    
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

COURSE_MAP: dict[str, dict] = {}
all_codes = set(LEGACY_COURSE_MAP.keys()) | set(_partner_map.keys())
for code in sorted(all_codes):
    partner_entry = _partner_map.get(code)
    legacy_entry = LEGACY_COURSE_MAP.get(code)
    merged = {
        "name": (legacy_entry or {}).get("name") or code,
        "code": code,
        "group": (partner_entry or {}).get("group") if partner_entry and partner_entry.get("group") is not None else (legacy_entry or {}).get("group"),
        "full_name": (partner_entry or {}).get("full_name") or (legacy_entry or {}).get("name") or code,
    }
    COURSE_MAP[code] = merged

# Corrected: PRTC201 is not visible by default
DEFAULT_VISIBLE_COURSES = {"AIML101", "PRTC301"}

# --- Other Settings ---
# Cutoff date logic handled dynamically in the pipeline


# --- Structured settings (via Pydantic Settings) ---
class HarvesterSettings(BaseSettings):
    """Centralized, typed settings for harvester behavior.

    Values can be configured via environment variables. Prefer the
    HARVESTER_* variants, but common legacy envs are supported where noted.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Secrets loaded from .env ---
    # These fields are added to accept the values from the .env file
    # created by the GitHub Action, resolving the ValidationError.
    GEMINI_API_KEY: str = Field(default="...")
    COHERE_API_KEY: str = Field(default="...")
    SUPABASE_URL: str = Field(default="...")
    SUPABASE_KEY: str = Field(default="...")
    COACH_USERNAME: str = Field(default="...")
    COACH_PASSWORD: str = Field(default="...")

    # --- Harvester-specific settings ---
    # Accept both HARVESTER_SELENIUM_HEADLESS and legacy SELENIUM_HEADLESS
    selenium_headless: bool = Field(
        default=True,
        validation_alias=AliasChoices("HARVESTER_SELENIUM_HEADLESS", "SELENIUM_HEADLESS"),
    )

    # Page load timeout for Selenium's driver
    page_load_timeout: int = Field(
        default=60,
        validation_alias=AliasChoices("HARVESTER_PAGE_LOAD_TIMEOUT"),
    )

    # Default wait timeout for WebDriverWait operations
    wait_timeout: int = Field(
        default=30,
        validation_alias=AliasChoices("HARVESTER_WAIT_TIMEOUT"),
    )

    # Directory to store screenshots when errors occur
    screenshot_dir: str = Field(
        default="logs/error_screenshots",
        validation_alias=AliasChoices("HARVESTER_SCREENSHOT_DIR"),
    )

    # Temporary downloads directory for any saved assets
    downloads_dir: str = Field(
        default="/tmp/harvester_downloads",
        validation_alias=AliasChoices("HARVESTER_DOWNLOADS_DIR"),
    )


# Instantiate settings once for module-level access
SETTINGS = HarvesterSettings()