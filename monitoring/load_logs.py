"""
load_logs.py
-----------------------
Utility functions for loading and querying RAG monitoring logs.

Reads:
    logs/rag_requests.jsonl

Provides:
    • load_all_logs()
    • get_last_n(n)
    • filter_by_date(start, end)
    • filter_low_grounding(threshold)
    • filter_high_latency(ms)
    • filter_by_query_substring(text)

All functions return Python lists of dictionaries.
"""

import json
from pathlib import Path
from datetime import datetime


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "rag_requests.jsonl"


# --------------------------------------------------------------------
# Load all log entries
# --------------------------------------------------------------------
def load_all_logs():
    """Load the entire JSONL log file as a list of dicts."""
    if not LOG_FILE.exists():
        return []

    events = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip corrupted or partial lines
                continue
    return events


# --------------------------------------------------------------------
# Get last N log entries
# --------------------------------------------------------------------
def get_last_n(n: int):
    """Return the most recent N log entries."""
    logs = load_all_logs()
    return logs[-n:] if n > 0 else []


# --------------------------------------------------------------------
# Filter logs by date range
# --------------------------------------------------------------------
def filter_by_date(start: str, end: str):
    """
    Filter logs between two ISO timestamps.

    Example:
        filter_by_date("2025-11-24T00:00", "2025-11-25T00:00")
    """
    logs = load_all_logs()

    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except Exception:
        return []  # invalid date format → return empty

    results = []
    for entry in logs:
        ts = entry.get("timestamp")
        if not ts:
            continue

        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue

        if start_dt <= dt <= end_dt:
            results.append(entry)

    return results


# --------------------------------------------------------------------
# Filter by grounding score
# --------------------------------------------------------------------
def filter_low_grounding(threshold: float = 0.5):
    """Return entries where grounding_score < threshold."""
    logs = load_all_logs()
    return [
        e for e in logs
        if float(e.get("grounding_score", 0.0)) < threshold
    ]


# --------------------------------------------------------------------
# Filter high-latency events
# --------------------------------------------------------------------
def filter_high_latency(ms: float = 3000.0):
    """Return entries where latency_ms > ms."""
    logs = load_all_logs()
    return [
        e for e in logs
        if float(e.get("latency_ms", 0.0)) > ms
    ]


# --------------------------------------------------------------------
# Filter logs containing part of a query
# --------------------------------------------------------------------
def filter_by_query_substring(text: str):
    """
    Match logs where the query contains a given substring.
    Case-insensitive.
    """
    logs = load_all_logs()
    text = text.lower()

    return [
        e for e in logs
        if text in e.get("query", "").lower()
    ]


# --------------------------------------------------------------------
# Manual Test
# --------------------------------------------------------------------
if __name__ == "__main__":
    print("Total logs:", len(load_all_logs()))
    print("Last 2 logs:", get_last_n(2))
