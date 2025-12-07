"""
checks.py — Phase 8 Drift & Alert Checks
-----------------------------------------

Reusable low-level monitoring checks for:
    • latency anomalies
    • grounding drift
    • retrieval drift (confidence + semantic/topic drift)
    • IDK consistency for unknown questions
    • volume anomalies (request throughput)
    
Reads logs via:
    monitoring.load_logs.load_all_logs()

This module does NOT write files.
It only computes metrics and flags for alerts.py.
"""

import statistics
from datetime import datetime, timedelta
from collections import Counter
from typing import Dict, List, Tuple

from monitoring.load_logs import load_all_logs


# ---------------------------------------------------------
# Helper: safe percentile
# ---------------------------------------------------------
def _percentile(values: List[float], pct: float):
    if not values:
        return 0.0
    k = (len(values) - 1) * pct
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


# ---------------------------------------------------------
# Latency checks
# ---------------------------------------------------------
def check_latency_anomalies(
    threshold_ms: float = 3000.0,
    pct95_limit: float = 2500.0
) -> Dict:
    """
    Flags:
        • high_latency_count: number of requests above threshold_ms
        • p95_exceeded: True if P95 > pct95_limit
    """
    logs = load_all_logs()
    if not logs:
        return {"high_latency_count": 0, "p95_exceeded": False}

    latencies = [float(e.get("latency_ms", 0.0)) for e in logs]
    high_latency = [l for l in latencies if l > threshold_ms]

    lat_sorted = sorted(latencies)
    p95 = _percentile(lat_sorted, 0.95)

    return {
        "high_latency_count": len(high_latency),
        "p95_exceeded": p95 > pct95_limit,
        "p95_value": p95,
        "max_latency": max(latencies, default=0.0),
    }


# ---------------------------------------------------------
# Grounding checks
# ---------------------------------------------------------
def check_grounding_drift(expected_min: float = 1.0) -> Dict:
    """
    Since strict literal generator ALWAYS produces grounding_score=1.0,
    any deviation indicates a major issue.
    """
    logs = load_all_logs()
    if not logs:
        return {"grounding_drift": False, "min_grounding": 1.0}

    scores = [float(e.get("grounding_score", 0.0)) for e in logs]
    min_score = min(scores) if scores else 1.0

    return {
        "grounding_drift": min_score < expected_min,
        "min_grounding": min_score,
    }


# ---------------------------------------------------------
# Retrieval drift — Option A: Confidence Drift
# ---------------------------------------------------------
def check_confidence_drift(
    baseline_confidence: float = 0.70,
    tolerance: float = 0.15
) -> Dict:
    """
    Detects drop in average retrieval confidence.

    baseline_confidence:
        Derived from Phase 6 evaluation (you can adjust)
    tolerance:
        Allowed deviation before alert triggers
    """
    logs = load_all_logs()
    if not logs:
        return {"confidence_drift": False, "avg_confidence": 0.0}

    confs = [float(e.get("confidence", 0.0)) for e in logs]
    avg_conf = statistics.mean(confs)

    drift = avg_conf < (baseline_confidence - tolerance)

    return {
        "confidence_drift": drift,
        "avg_confidence": avg_conf,
        "baseline_confidence": baseline_confidence,
    }


# ---------------------------------------------------------
# Retrieval drift — Option C: Topic / Chunk Drift
# ---------------------------------------------------------
def check_semantic_drift(
    baseline_topics: List[str] = None,
    drift_ratio: float = 0.30
) -> Dict:
    """
    Detects semantic/topic drift by comparing chunk topics across logs.
    
    baseline_topics:
        Optional list of expected topics (from Phase 6)
        If None → we use the most common topics from logs
    
    drift_ratio:
        Maximum allowed change in topic distribution
    
    Returns:
        {
            'semantic_drift': bool,
            'top_topics': [...],
            'topic_distribution': {...},
        }
    """
    logs = load_all_logs()
    if not logs:
        return {"semantic_drift": False, "top_topics": [], "topic_distribution": {}}

    # Extract topic tags from retrieved[0]['topic'] if present
    topics = []
    for e in logs:
        retrieved = e.get("retrieved", [])
        if retrieved and isinstance(retrieved[0], dict):
            topics.append(retrieved[0].get("topic", "unknown"))
        else:
            topics.append("unknown")

    if not topics:
        return {"semantic_drift": False, "top_topics": [], "topic_distribution": {}}

    counter = Counter(topics)
    total = sum(counter.values())
    dist = {k: v / total for k, v in counter.items()}

    # If no baseline provided, set the current distribution as stable baseline
    if baseline_topics is None:
        baseline_topics = list(counter.keys())

    # Drift if new topics dominate or distribution changes radically
    unexpected_topics = [t for t in dist if t not in baseline_topics]

    semantic_drift = False
    if unexpected_topics:
        unexpected_ratio = sum(dist[t] for t in unexpected_topics)
        semantic_drift = unexpected_ratio > drift_ratio

    return {
        "semantic_drift": semantic_drift,
        "top_topics": baseline_topics,
        "topic_distribution": dist,
        "unexpected_topics": unexpected_topics,
    }


# ---------------------------------------------------------
# IDK Consistency Checks
# ---------------------------------------------------------
def check_idk_behavior() -> Dict:
    """
    Unknown-type questions should return:
        "I don't know."
    
    This checks consistency of fallback behavior.
    
    NOTE:
    This does NOT classify questions as known/unknown.
    It simply looks for answers that are EXACTLY IDK
    when grounding/context == 0.
    """
    logs = load_all_logs()
    if not logs:
        return {"idk_violations": 0}

    violations = 0
    for e in logs:
        used = e.get("used_chunks", [])
        answer = e.get("answer", "").strip().lower()

        if not used:  # fallback situation
            if answer not in {"i don't know.", "i dont know.", "i do not know."}:
                violations += 1

    return {"idk_violations": violations}


# ---------------------------------------------------------
# Volume / throughput anomalies
# ---------------------------------------------------------
def check_volume_anomalies(
    window_minutes: int = 60,
    max_expected_per_minute: float = 30.0
) -> Dict:
    """
    Detect sudden spikes in request volume.
    """
    logs = load_all_logs()
    if not logs:
        return {"volume_alert": False, "requests_last_hour": 0}

    now = datetime.utcnow()
    window_start = now - timedelta(minutes=window_minutes)

    recent = [
        e for e in logs
        if "timestamp" in e
        and datetime.fromisoformat(e["timestamp"]) >= window_start
    ]

    rpm = len(recent) / window_minutes  # requests per minute
    volume_alert = rpm > max_expected_per_minute

    return {
        "volume_alert": volume_alert,
        "requests_last_hour": len(recent),
        "rpm": rpm,
    }
