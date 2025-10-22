# src/shared/utils.py

import re
import datetime
import logging
from typing import Optional # For type hinting

def parse_session_date(date_str: str) -> Optional[datetime.datetime]:
    """
    Parses date strings like 'DD/MM/YYYY' possibly followed by '| Time'.
    Returns a datetime object (UTC assumed for consistency) or None if parsing fails.
    """
    if not date_str:
        return None
        
    date_part = date_str.split('|')[0].strip()
    # Handle potential extra text sometimes found
    date_part = date_part.split(' ')[0].strip() 
    
    try:
        # Specify the expected format
        dt_obj = datetime.datetime.strptime(date_part, '%d/%m/%Y')
        # Standardize to UTC timezone for consistency
        return dt_obj.replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        logging.warning(f"Could not parse date string: '{date_str}' with format DD/MM/YYYY")
        # Try other common formats if necessary, e.g., MM/DD/YYYY
        # try:
        #     dt_obj = datetime.datetime.strptime(date_part, '%m/%d/%Y')
        #     return dt_obj.replace(tzinfo=datetime.timezone.utc)
        # except ValueError:
        #     logging.error(f"Failed to parse date string '{date_str}' with known formats.")
        #     return None
        return None # Return None if primary format fails

def create_safe_filename(text: str, max_length: int = 100) -> str:
    """
    Cleans a string to make it suitable for use as a filename.
    Removes special characters, replaces spaces, and truncates length.
    """
    if not text:
        return "untitled"

    # Remove potential file path characters and other problematic symbols
    # Keep alphanumeric, underscores, hyphens
    text = re.sub(r'[^\w\-]+', '_', text)

    # Replace multiple consecutive underscores with a single one
    text = re.sub(r'_+', '_', text)

    # Remove leading/trailing underscores
    text = text.strip('_')

    # Truncate to max_length (leaving space for extension if added later)
    return text[:max_length]

# --- Example Usage (can be removed or kept under if __name__ == "__main__": block) ---
if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test date parsing
    date1 = "17/10/2025 | 10:00 AM"
    date2 = "21/11/2025"
    date3 = "Invalid Date"
    date4 = "10-25-2025" # Example of format not currently handled
    print(f"'{date1}' -> {parse_session_date(date1)}")
    print(f"'{date2}' -> {parse_session_date(date2)}")
    print(f"'{date3}' -> {parse_session_date(date3)}")
    print(f"'{date4}' -> {parse_session_date(date4)}")

    # Test filename sanitization
    title1 = "Lecture 5: Confidence Intervals & Z-Scores!"
    title2 = "   Case Study / Market Analysis --- (Part 1)"
    title3 = ""
    print(f"'{title1}' -> '{create_safe_filename(title1)}'")
    print(f"'{title2}' -> '{create_safe_filename(title2)}'")
    print(f"'{title3}' -> '{create_safe_filename(title3)}'")