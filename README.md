# 🎓 Professor AI Tutor RAG System

This project is an advanced Retrieval-Augmented Generation (RAG) system designed to provide AI-powered tutoring and study guidance based *strictly* on specific course materials.

It features a robust, automated pipeline (Harvester + Refinery) to ingest lecture transcripts, PDFs, and web content from a partner learning platform, and a secure, cost-effective FastAPI backend that serves student queries.

## 🏗️ Architecture Overview

The system is split into three main components:

1.  **Harvester (`src/harvester/`):** A fully automated **Selenium** web scraper managed by GitHub Actions. It logs into the platform, navigates to courses, and downloads/scrapes all new resources.
2.  **Refinery (`src/refinery/`):** Processes raw data. It uses the Gemini LLM for cleaning transcripts (removing timestamps, fixing punctuation) and vision analysis for describing images in PDFs. It then chunks and embeds the data into a **Supabase** vector store.
3.  **Application (FastAPI):** A secure **FastAPI** server (`api_server.py`) that handles RAG queries. It uses zero-cost **RRF/MMR re-ranking** to retrieve the most relevant and diverse context before generating an answer using the Gemini API.

## ✨ Key Features

* **Zero-Cost RAG:** Uses a custom RRF (Reciprocal Rank Fusion) and MMR (Maximal Marginal Relevance) algorithm in place of expensive third-party re-ranking APIs (like Cohere).
* **Multi-Modal Content:** Processes text, links, and uses Gemini Vision to describe images within PDF pages.
* **Automated Pipeline:** The entire data ingestion process is managed by a daily GitHub Actions workflow.
* **Security:** API endpoints are secured with a mandatory API Key (`SECRET_API_KEY`) and rate limiting (`slowapi`).

### Manual Transcript Ingestion (Fallback for Scraping Issues)

If the automated Drive scraping fails, manually process transcript files:

```bash
# 1. Place .txt files in the manual directory
# Format: {ClassName}.txt (e.g., Statistics.txt)

# 2. Run the ingestion script
python scripts/ingest_manual.py

# 3. Check logs for detailed processing information
tail -f logs/manual_ingest.log

# 4. Optional: Validate file formats before processing
python scripts/ingest_manual.py --validate-only

# 5. Optional: Test run without database writes
python scripts/ingest_manual.py --dry-run
```

**File Format:**
```
ClassName - Lecture Title
Date: YYYY-MM-DD | Time: HH:MM AM/PM

[Transcript content starting from line 4...]
0:01 Welcome to the first lecture...
5:23 Today we'll cover normal distributions...
```

**Key Features:**
- **De-duplication:** Automatically skips files already processed
- **Flexible Date Parsing:** Handles various date formats (YYYY-MM-DD, DD/MM/YYYY, "January 15, 2025")
- **LLM Cleaning:** Uses Gemini to remove timestamps and improve readability
- **Error Recovery:** Continues processing other files if one fails
- **Comprehensive Logging:** Detailed logs for debugging and monitoring

## ⚙️ Local Setup and Installation

### Prerequisites

* Python 3.11+
* **Supabase Project:** With the `vector` extension enabled and the `documents` table created.
* **Google AI Studio Key:** For Gemini LLM and Embeddings.

### 1. Clone and Install

```bash
# Clone the repository
git clone [Your Repository URL]
cd professor-agent-platform

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (this includes FastAPI, Selenium, Pydantic, etc.)
pip install -r requirements.txt