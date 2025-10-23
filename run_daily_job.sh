#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting Daily Pipeline..."
echo "================================"

# Load environment variables from .env file
if [ -f .env ]; then
    echo "✅ Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
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