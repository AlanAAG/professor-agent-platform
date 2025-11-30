# debug_gemini.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load .env
load_dotenv()

# 2. Check Variables
gemini_key = os.getenv("GEMINI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

print(f"--- Environment Check ---")
if gemini_key:
    print(f"✅ GEMINI_API_KEY found: {gemini_key[:6]}...{gemini_key[-4:]}")
else:
    print(f"❌ GEMINI_API_KEY is NOT found.")

if google_key:
    print(f"✅ GOOGLE_API_KEY found: {google_key[:6]}...{google_key[-4:]}")
else:
    print(f"❌ GOOGLE_API_KEY is NOT found.")

# 3. Test the API connection directly (bypassing LangChain)
active_key = google_key or gemini_key
if not active_key:
    print("❌ No key found to test!")
    exit(1)

print(f"\n--- Connecting to Google Gemini ---")
try:
    genai.configure(api_key=active_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello, are you working?")
    print(f"✅ SUCCESS! Gemini replied: {response.text}")
except Exception as e:
    print(f"❌ ERROR: {e}")