"""
analyze_logs.py
-------------------------------------------------------
Analytics toolkit for Phase 7 RAG Monitoring.

Uses logs from:
    logs/rag_requests.jsonl

Provides:
    • summarize_logs()
    • top_queries(n)
    • latency_stats()
    • grounding_stats()
    • detect_hallucinations(threshold)
    • print_report()

Dependencies:
    • load_logs.py (same folder)
"""

import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List

from monitoring.load_logs import load_all_logs


# --------------------------------------------------------------------
# Summary utilities
# --------------------------------------------------------------------
def summarize_logs() -> Dict:
    """Return high-level summary of all logs."""
    logs = load_all_logs()
    if not logs:
        return {"total_logs": 0}

    total = len(logs)

    avg_latency = statistics.mean([e.get("latency_ms", 0.0) for e in logs])
    avg_grounding = statistics.mean([e.get("grounding_score", 0.0) for e in logs])

    return {
        "total_logs": total,
        "avg_latency_ms": round(avg_latency, 3),
        "avg_grounding_score": round(avg_grounding, 4),
    }


# --------------------------------------------------------------------
# Query frequency
# --------------------------------------------------------------------
def top_queries(n: int = 10) -> List:
    """Return the top N most frequent queries."""
    logs = load_all_logs()
    queries = [e.get("query", "").strip().lower() for e in logs]

    counter = Counter(queries)
    return counter.most_common(n)


# --------------------------------------------------------------------
# Latency analytics
# --------------------------------------------------------------------
def latency_stats() -> Dict:
    """Return min/avg/median/max latency."""
    logs = load_all_logs()
    latencies = [e.get("latency_ms", 0.0) for e in logs]

    if not latencies:
        return {}

    return {
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "median_ms": statistics.median(latencies),
        "avg_ms": statistics.mean(latencies),
    }


# --------------------------------------------------------------------
# Grounding score analytics
# --------------------------------------------------------------------
def grounding_stats() -> Dict:
    """Return distribution metrics for grounding score."""
    logs = load_all_logs()
    gs = [e.get("grounding_score", 0.0) for e in logs]

    if not gs:
        return {}

    return {
        "min": round(min(gs), 4),
        "max": round(max(gs), 4),
        "median": round(statistics.median(gs), 4),
        "avg": round(statistics.mean(gs), 4),
    }


# --------------------------------------------------------------------
# Hallucination detection
# --------------------------------------------------------------------
def detect_hallucinations(threshold: float = 0.50) -> List[Dict]:
    """
    Return log entries where grounding_score < threshold.
    These are considered likely hallucinations.
    """
    logs = load_all_logs()

    hallucinations = [
        e for e in logs
        if float(e.get("grounding_score", 0.0)) < threshold
    ]
    return hallucinations


# --------------------------------------------------------------------
# Pretty-printed full report
# --------------------------------------------------------------------
def print_report():
    """Print a full analytics summary to console."""
    logs = load_all_logs()
    print("\n==================== RAG Monitoring Report ====================\n")

    if not logs:
        print("No logs found.\n")
        return

    # High-level summary
    summary = summarize_logs()
    print("Total Logged Requests:", summary["total_logs"])
    print("Average Latency (ms):", summary["avg_latency_ms"])
    print("Average Grounding Score:", summary["avg_grounding_score"], "\n")

    # Top queries
    print("Top Queries:")
    for q, count in top_queries(5):
        print(f"  • {q} ({count} times)")
    print()

    # Latency stats
    lat = latency_stats()
    print("Latency Stats (ms):")
    for k, v in lat.items():
        print(f"  {k}: {round(v, 3)}")
    print()

    # Grounding stats
    gs = grounding_stats()
    print("Grounding Score Stats:")
    for k, v in gs.items():
        print(f"  {k}: {v}")
    print()

    # Hallucinations
    hall = detect_hallucinations()
    print(f"Possible Hallucinations (< 0.5 grounding): {len(hall)}")
    print("==============================================================\n")


# --------------------------------------------------------------------
# Manual test
# --------------------------------------------------------------------
if __name__ == "__main__":
    print_report()
