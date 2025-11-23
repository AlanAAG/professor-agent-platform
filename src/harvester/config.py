"""
Selenium-oriented configuration for the harvester module.

This converts prior Playwright selectors into robust Selenium-friendly
XPaths/CSS selectors and keeps course mappings and URLs separated from logic.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from selenium.webdriver.common.by import By
import os

# --- Core URLs ---
LOGIN_URL = "https://coach.tetr.com/login"
BASE_URL = "https://coach.tetr.com/"
COURSES_URL = BASE_URL + "courses"

# File path to persist cookies/session if desired (used by Selenium helpers)
AUTH_STATE_FILE = "data/auth_state.json"

# --- Login Page Selectors (Selenium) ---
# Prioritized selector lists to support resilient fallbacks.
USERNAME_SELECTORS = [
    (By.XPATH, "//input[@placeholder='Enter Your Email ID']"),
    (By.XPATH, "//input[@name='officialEmail']"),
]

PASSWORD_SELECTORS = [
    (By.XPATH, "//input[@placeholder='Enter Your Password']"),
    (By.XPATH, "//input[@name='password']"),
]

LOGIN_BUTTON_SELECTORS = [
    (By.XPATH, "//button[normalize-space(.)='Login']"),
    (By.XPATH, "//button[normalize-space(.)='Sign In']"),
    (By.XPATH, "//button[@type='submit']"),
]

# A generic indicator that we are not on the login page anymore
DASHBOARD_INDICATOR_CSS = '#gtm-IdDashboard'

# --- Courses Page (Selenium) ---
# Group header: div.domainHeader containing p.title == {group_name}
GROUP_HEADER_XPATH_TEMPLATE = "//p[normalize-space(.)='{group_name}']/ancestor::div[contains(@class, 'domainHeader')][1]"

# Course link: anchor whose visible identifier matches the course code within a stable container
COURSE_LINK_XPATH_TEMPLATE = "//span[normalize-space(.)='{course_code}']/ancestor::a[1]"

# Fallback course card container (handles layouts where program code sits inside divs under anchor wrappers)
COURSE_CARD_FALLBACK_XPATH_TEMPLATE = (
    "//span[contains(concat(' ', normalize-space(@class), ' '), ' pIdName ') and normalize-space(.)='{course_code}']"
    "/ancestor::div[contains(@class, 'sc-eDWCr') or contains(@class, 'domainCourses')][1]"
)

# --- Course Details Page (Resources Tab Navigation) ---
RESOURCES_TAB_SELECTORS = [
    (
        By.XPATH,
        "//div[.//img[contains(@src, 'resources.svg')] and .//p[normalize-space(.)='Resources']]",
    ),
    (
        By.XPATH,
        "//p[normalize-space(.)='Resources']/ancestor::div[contains(@class, 'sc-kMjNwy')][1]",
    ),
    (
        By.XPATH,
        "//p[normalize-space(.)='Resources']/ancestor::li[1]",
    ),
]

# Section headers inside resources
SECTION_HEADER_XPATH_TPL = "//p[contains(@class, 'name') and normalize-space(.)='{section_title}']"

PRE_READ_SECTION_TITLE = "Pre-Read Materials"
IN_CLASS_SECTION_TITLE = "In Class Materials"
POST_CLASS_SECTION_TITLE = "Post Class Materials"
SESSION_RECORDINGS_SECTION_TITLE = "Session Recordings"

# Resource items and sub-elements
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
    "AIML101": {"name": "AIML", "group": "Quantitative Tools for Business"},
    "PRTC301": {"name": "Statistics", "group": "Quantitative Tools for Business"},
    "PRTC201": {"name": "Excel", "group": "Management Accounting"},
    "CAL101": {"name": "Calculus", "group": "Mathematics for Engineers"},
    "MAST401": {"name": "Startup", "group": "Management Project - I"},
    "CAP101": {"name": "Dropshipping", "group": "Management Project - I"},
    "COMM101": {"name": "PublicSpeaking", "group": "Management Project - I"},
    "MAST601": {"name": "Networking", "group": "Management Project - I"},
    "CS101": {"name": "OOP", "group": "Computer Science"},
    "FIFI101": {"name": "FinanceBasics", "group": "Management Accounting"},
    "MAST102": {"name": "MarketAnalysis", "group": "Management Accounting"},
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

DEFAULT_VISIBLE_COURSES = {"AIML101", "PRTC301"}

class HarvesterSettings(BaseSettings):
    """Centralized, typed settings for harvester behavior."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    selenium_headless: bool = Field(
        default=True,
        validation_alias=AliasChoices("HARVESTER_SELENIUM_HEADLESS", "SELENIUM_HEADLESS"),
    )

    page_load_timeout: int = Field(
        default=60,
        validation_alias=AliasChoices("HARVESTER_PAGE_LOAD_TIMEOUT"),
    )

    wait_timeout: int = Field(
        default=30,
        validation_alias=AliasChoices("HARVESTER_WAIT_TIMEOUT"),
    )

    screenshot_dir: str = Field(
        default="logs/error_screenshots",
        validation_alias=AliasChoices("HARVESTER_SCREENSHOT_DIR"),
    )

    downloads_dir: str = Field(
        default="/tmp/harvester_downloads",
        validation_alias=AliasChoices("HARVESTER_DOWNLOADS_DIR"),
    )
    
    resource_batch_size: int = Field(
        default=50,
        validation_alias=AliasChoices("HARVESTER_RESOURCE_BATCH_SIZE"),
    )

    telemetry_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("HARVESTER_TELEMETRY_ENABLED", "TELEMETRY_ENABLED"),
    )

    telemetry_log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("HARVESTER_TELEMETRY_LOG_LEVEL", "TELEMETRY_LOG_LEVEL"),
    )

    metrics_report_path: str = Field(
        default="data/pipeline_status.json",
        validation_alias=AliasChoices("HARVESTER_METRICS_REPORT_PATH", "METRICS_REPORT_PATH"),
    )

    # --- NEW: Gemini & Fallback Settings (Replaces Whisper) ---
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )

    # Generic fallback flag (supports legacy env var name for backward compatibility)
    enable_recording_fallback: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_RECORDING_FALLBACK", "ENABLE_WHISPER_FALLBACK"),
    )

    # Generic download limit (supports legacy env var name)
    recording_max_download_mb: float = Field(
        default=400.0,
        validation_alias=AliasChoices("RECORDING_MAX_DOWNLOAD_MB", "WHISPER_MAX_DOWNLOAD_MB"),
    )

SETTINGS = HarvesterSettings()