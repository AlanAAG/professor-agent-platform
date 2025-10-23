#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting Daily Pipeline..."
echo "================================"

# Load environment variables from .env file (robustly handles spaces/quotes)
if [ -f .env ]; then
    echo "✅ Loading environment variables from .env"
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
else
    echo "❌ Error: .env file not found!"
    exit 1
fi

# Ensure we're in the project root
cd "$(dirname "$0")"

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p logs/error_screenshots
mkdir -p data/raw_transcripts
mkdir -p data/raw_static
mkdir -p data/raw_pdfs
mkdir -p /tmp/harvester_downloads

# Set pipeline mode (default to 'daily', can override with argument)
PIPELINE_MODE=${1:-daily}
export PIPELINE_MODE

echo "📊 Pipeline mode: $PIPELINE_MODE"
echo "================================"
echo ""

# Ensure Playwright browsers are installed (idempotent)
echo "🧭 Ensuring Playwright browsers are installed..."
python -m playwright install --with-deps chromium >/dev/null 2>&1 || true

# Run the pipeline
python -m src.run_hybrid_pipeline

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✅ Pipeline completed successfully!"
else
    echo ""
    echo "================================"
    echo "❌ Pipeline failed!"
    exit 1
fi