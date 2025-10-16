import os
import asyncio
from dotenv import load_dotenv
from playwright.async_api import async_playwright

# Load environment variables from the .env file
load_dotenv()

# --- Configuration ---
LOGIN_URL = "https://coach.tetr.com/"
AUTH_STATE_FILE = "auth_state.json"

# Retrieve credentials securely from environment variables
USERNAME = os.environ.get("COACH_USERNAME")
PASSWORD = os.environ.get("COACH_PASSWORD")

async def perform_login(page):
    """
    Handles the login process for the Coach platform.
    """
    print(f"-> Navigating to: {LOGIN_URL}")
    await page.goto(LOGIN_URL)

    print("-> Attempting login...")
    try:
        # Use confirmed input names for credentials
        await page.fill('input[name="officialEmail"]', USERNAME)
        await page.fill('input[name="password"]', PASSWORD)
        
        # Click the confirmed login button ID
        await page.click('#gtmLoginStd')

        # --- Dashboard/Post-Login Check (Step 3 Setup) ---
        try:
            DASHBOARD_SELECTOR = '#gtm-IdDashboard'
            await page.wait_for_selector(DASHBOARD_SELECTOR, state="visible", timeout=20000)
            
            print(f"✅ Login Successful! Dashboard element ({DASHBOARD_SELECTOR}) found.")
            
            # Save authentication state ONLY ONCE upon success
            await page.context.storage_state(path=AUTH_STATE_FILE)
            print(f"🔑 Authentication state saved to {AUTH_STATE_FILE}.")
            return True

        except Exception:
            # This handles failure to load the dashboard element
            print("❌ Login Failed: Dashboard element not found after 20 seconds. Check credentials/network.")
            await page.screenshot(path="login_error_dashboard_fail.png")
            return False

    except Exception as e:
        # This handles failure to find the initial login form elements (fill/click)
        print(f"❌ Initial Login Form Failed (Timeout or Selector Error): {e}")
        await page.screenshot(path="login_error_form_fail.png")
        return False

    return True # Should not be reached, but good practice

async def main():
    if not USERNAME or not PASSWORD:
        print("ERROR: COACH_USERNAME or COACH_PASSWORD not set in .env or environment variables.")
        return
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        context = await browser.new_context()
        page = await context.new_page()

        await perform_login(page)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())