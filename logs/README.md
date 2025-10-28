# Pipeline Logs

## Run Summaries

Each pipeline run generates two summary files:
- `run_summary_YYYYMMDD_HHMMSS.json` - Timestamped historical record
- `run_summary_latest.json` - Always points to most recent run

### Summary Structure
```json
{
  "run_metadata": {
    "timestamp": "2025-01-15T10:30:00",
    "mode": "daily",
    "duration_seconds": 245.67,
    "duration_human": "4m 5s",
    "success": true
  },
  "statistics": {
    "courses_attempted": 13,
    "courses_successful": 13,
    "resources_discovered": 47,
    "resources_processed": 45,
    "resources_failed": 2
  },
  "errors": [
    {
      "resource_title": "Week 3 Slides",
      "error_type": "TimeoutException",
      "error_message": "Element not found...",
      "timestamp": "2025-01-15T10:32:15"
    }
  ],
  "error_count": 2
}
```

## Quick Analysis

### Check Latest Run Status
```bash
cat logs/run_summary_latest.json | jq '.run_metadata.success'
```

### Count Errors in Last 7 Runs
```bash
cat logs/run_summary_*.json | jq -s 'map(.error_count) | add'
```

### Find Most Common Errors
```bash
cat logs/run_summary_*.json | jq -s 'map(.errors[]) | group_by(.error_type) | map({type: .[0].error_type, count: length}) | sort_by(.count) | reverse'
```

## Error Log

Detailed error logs with stack traces: `pipeline.log`

Error screenshots (if any): `error_screenshots/`
