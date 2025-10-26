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

# Resolve Python interpreter
if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

# Install dependencies (idempotent)
echo "📦 Installing Python dependencies..."
$PYTHON -m pip install -r requirements.txt >/dev/null 2>&1 || {
  echo "❌ Pip install failed";
  exit 1;
}

# Ensure Playwright browsers are installed (idempotent)
echo "🧭 Ensuring Playwright browsers are installed..."
# Allow switching engine via PW_ENGINE (chromium|firefox|webkit)
ENGINE=${PW_ENGINE:-chromium}
$PYTHON -m playwright install --with-deps "$ENGINE" >/dev/null 2>&1 || true

# Run the pipeline (use xvfb-run if headful and available)
if [ "${PW_HEADLESS,,}" = "false" ] && command -v xvfb-run >/dev/null 2>&1; then
  echo "🖥️  Running headful under Xvfb..."
  xvfb-run -a $PYTHON -m src.run_hybrid_pipeline
else
  $PYTHON -m src.run_hybrid_pipeline
fi

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