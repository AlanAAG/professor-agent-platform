# src/refinery/document_processing.py
import logging
from typing import Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


def extract_drive_content(driver: webdriver.Chrome, url: str, doc_type: str) -> Optional[str]:
    """
    Navigates to the Google Drive/Office Web viewer URL and scrapes the visible text content.
    This avoids downloading the binary file (PPTX, DOCX, XLSX).
    """
    logging.info(f"Attempting in-browser content extraction for {doc_type} at: {url}")

    # 1. Navigate to the document viewer page
    driver.get(url)

    # Define a general CSS selector for primary text content containers
    # Use a generic, robust selector set as a starting point; broaden as needed with testing.
    COMMON_CONTENT_SELECTORS = [
        "div[role='main']",   # Common for main document content
        ".kix-page",          # Google Docs pages
        ".kix-appview-editor",  # Google Docs editor root (view-only renders text within)
        ".punch-viewer-page", # Google Slides pages
        "div.docs-sheet-tab-panel",  # Google Sheets visible grid container
    ]

    visible_content: list[str] = []

    try:
        # Use explicit wait to confirm at least one content container appears
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, COMMON_CONTENT_SELECTORS[0]))
        )

        # 2. Extract visible text using multiple selectors as fallback
        for selector in COMMON_CONTENT_SELECTORS:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed():
                        text = element.text
                        if text:
                            visible_content.append(text)
            except Exception:
                continue  # Try the next selector

        if visible_content:
            # Join with spacing to separate blocks
            return "\n\n".join(chunk.strip() for chunk in visible_content if chunk and chunk.strip())
        else:
            logging.warning(
                f"Could not find readable content in the web viewer for {doc_type}."
            )
            return None

    except TimeoutException:
        logging.error(
            f"Timed out waiting for viewer elements for {doc_type} at {url[:80]}..."
        )
        return None
    except Exception as e:
        logging.error(
            f"Error during in-browser content extraction for {doc_type}: {e}"
        )
        return None
