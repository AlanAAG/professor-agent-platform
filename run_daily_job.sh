#!/bin/bash

# Exit on any error
set -e

# Track runtime
START_TIME=$(date +%s)

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
    # Allow missing .env file if running in GitHub Actions where secrets are injected directly
    echo "⚠️ Proceeding without .env (expecting secrets from environment/CI)"
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


# Ensure Google Chrome exists when running locally; Selenium Manager handles driver
if ! command -v google-chrome >/dev/null 2>&1 && ! command -v chrome >/dev/null 2>&1; then
  echo "🧭 Installing Google Chrome for Selenium (local dev)..."
  if [ -x "$(command -v apt)" ]; then
    sudo apt-get update -y >/dev/null 2>&1 || true
    sudo apt-get install -y wget gnupg >/dev/null 2>&1 || true
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg || true
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list >/dev/null
    sudo apt-get update -y >/dev/null 2>&1 || true
    sudo apt-get install -y google-chrome-stable >/dev/null 2>&1 || {
      echo "⚠️  Failed to install google-chrome-stable; continuing."
    }
  else
    echo "⚠️  Could not auto-install Chrome on this platform."
  fi
fi

# Run the pipeline (use xvfb-run if headful and available)
set +e
if [ "${PW_HEADLESS,,}" = "false" ] && command -v xvfb-run >/dev/null 2>&1; then
  echo "🖥️  Running headful under Xvfb..."
  xvfb-run -a $PYTHON -m src.run_hybrid_pipeline
  PIPELINE_EXIT=$?
else
  $PYTHON -m src.run_hybrid_pipeline
  PIPELINE_EXIT=$?
fi
set -e

END_TIME=$(date +%s)
RUNTIME_SECONDS=$((END_TIME - START_TIME))

# Default metrics
FINAL_STATUS="SUCCESS"
RESOURCES_PROCESSED="unknown"

# Prefer exit code; if non-zero, mark failure regardless of JSON
if [ "$PIPELINE_EXIT" -ne 0 ]; then
  FINAL_STATUS="FAILURE"
fi

# Try to read final summary JSON if present
STATUS_JSON_PATH="data/pipeline_status.json"
if [ -f "$STATUS_JSON_PATH" ] && [ -s "$STATUS_JSON_PATH" ]; then
  echo "📄 Found summary report at $STATUS_JSON_PATH"
  # Use Python for robust JSON parsing without external deps
  PY_PARSE=$(cat <<'PYEOF'
import json, os, sys
path = sys.argv[1]
try:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"status=UNKNOWN\nresources=unknown\nruntime=", end='')
    sys.exit(0)

status = (
    str(data.get('status') or data.get('final_status') or data.get('result') or '').upper()
)
resources = (
    data.get('resources_processed') or data.get('processed_count') or data.get('num_items')
    or data.get('documents_processed') or data.get('records_processed')
)
runtime = (
    data.get('runtime_seconds') or data.get('duration_seconds') or data.get('run_seconds')
    or data.get('duration')
)
print(f"status={status or 'UNKNOWN'}")
print(f"resources={resources if resources is not None else 'unknown'}")
print(f"runtime={runtime if runtime is not None else ''}", end='')
PYEOF
)
  PARSED=$(python -c "$PY_PARSE" "$STATUS_JSON_PATH" || true)
  JSON_STATUS=$(echo "$PARSED" | sed -n 's/^status=//p')
  RESOURCES_FROM_JSON=$(echo "$PARSED" | sed -n 's/^resources=//p')
  RUNTIME_FROM_JSON=$(echo "$PARSED" | sed -n 's/^runtime=//p')

  # Adopt JSON fields if available
  if [ -n "$JSON_STATUS" ] && [ "$JSON_STATUS" != "UNKNOWN" ]; then
    FINAL_STATUS="$JSON_STATUS"
  fi
  if [ -n "$RESOURCES_FROM_JSON" ]; then
    RESOURCES_PROCESSED="$RESOURCES_FROM_JSON"
  fi
  if [ -n "$RUNTIME_FROM_JSON" ]; then
    RUNTIME_SECONDS="$RUNTIME_FROM_JSON"
  fi
else
  echo "⚠️  Summary report not found at $STATUS_JSON_PATH"
fi

# Emit console summary
echo ""
echo "================================"
if [ "$FINAL_STATUS" = "SUCCESS" ]; then
  echo "✅ Pipeline completed successfully!"
else
  echo "❌ Pipeline failed!"
fi
echo "📈 Resources processed: $RESOURCES_PROCESSED"
echo "⏱️  Runtime (seconds): $RUNTIME_SECONDS"

# Write GitHub Actions outputs if available
if [ -n "$GITHUB_OUTPUT" ]; then
  {
    echo "status=$FINAL_STATUS"
    echo "resources_processed=$RESOURCES_PROCESSED"
    echo "runtime_seconds=$RUNTIME_SECONDS"
  } >> "$GITHUB_OUTPUT"
fi

# Write a nice Job Summary if available
if [ -n "$GITHUB_STEP_SUMMARY" ]; then
  {
    echo "### Daily Pipeline Status"
    echo "- **Status**: $FINAL_STATUS"
    echo "- **Resources processed**: $RESOURCES_PROCESSED"
    echo "- **Runtime (seconds)**: $RUNTIME_SECONDS"
  } >> "$GITHUB_STEP_SUMMARY"
fi

# Also surface as workflow commands for quick visibility
echo "::notice title=Pipeline Status::$FINAL_STATUS | Resources: $RESOURCES_PROCESSED | Runtime(s): $RUNTIME_SECONDS"

# Exit non-zero on failure to fail the workflow appropriately
if [ "$FINAL_STATUS" != "SUCCESS" ]; then
  exit 1
fi