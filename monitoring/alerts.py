"""
alerts.py — Phase 8 Log Analysis & Alert Engine
------------------------------------------------

This module:
    • Loads all request logs (rag_requests.jsonl)
    • Runs drift/anomaly checks from checks.py
    • Summarizes health metrics
    • Generates alert objects (severity + message)
    • Writes:
          logs/phase8_summary.json
          logs/alerts.json

This does NOT modify logs or backend behavior.
"""

import json
from pathlib import Path
from datetime import datetime

from monitoring.load_logs import load_all_logs
from monitoring.checks import (
    check_latency_anomalies,
    check_grounding_drift,
    check_confidence_drift,
    check_semantic_drift,
    check_idk_behavior,
    check_volume_anomalies,
)


# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
SUMMARY_FILE = LOG_DIR / "phase8_summary.json"
ALERTS_FILE = LOG_DIR / "alerts.json"

LOG_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------
# Build an alert object
# --------------------------------------------------------------------
def _make_alert(severity: str, msg: str, data=None):
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "severity": severity,           # "CRITICAL" | "WARNING" | "INFO"
        "message": msg,
        "data": data or {},
    }


# --------------------------------------------------------------------
# MAIN ALERT ENGINE
# --------------------------------------------------------------------
def run_alert_checks():
    logs = load_all_logs()
    summary = {
        "total_requests": len(logs),
        "generated_at": datetime.utcnow().isoformat(),
    }

    alerts = []

    if not logs:
        alerts.append(_make_alert("INFO", "No logs available. System idle."))
        _write_outputs(summary, alerts)
        return summary, alerts

    # ---------------------------------------------------------------
    # 1. Latency anomalies
    # ---------------------------------------------------------------
    latency = check_latency_anomalies()
    summary["latency"] = latency

    if latency["high_latency_count"] > 0:
        alerts.append(
            _make_alert(
                "WARNING",
                f"{latency['high_latency_count']} requests exceeded latency threshold.",
                latency,
            )
        )

    if latency["p95_exceeded"]:
        alerts.append(
            _make_alert(
                "CRITICAL",
                f"P95 latency {latency['p95_value']:.2f} ms exceeded healthy limit.",
                latency,
            )
        )

    # ---------------------------------------------------------------
    # 2. Grounding drift (strict literal mode → should remain 1.0)
    # ---------------------------------------------------------------
    grounding = check_grounding_drift()
    summary["grounding"] = grounding

    if grounding["grounding_drift"]:
        alerts.append(
            _make_alert(
                "CRITICAL",
                "Grounding score dropped below expected 1.0 — serious drift detected.",
                grounding,
            )
        )

    # ---------------------------------------------------------------
    # 3. Retrieval drift (A) — confidence drop
    # ---------------------------------------------------------------
    confidence = check_confidence_drift()
    summary["retrieval_confidence"] = confidence

    if confidence["confidence_drift"]:
        alerts.append(
            _make_alert(
                "CRITICAL",
                f"Average retrieval confidence dropped below baseline "
                f"({confidence['avg_confidence']:.3f}).",
                confidence,
            )
        )

    # ---------------------------------------------------------------
    # 4. Retrieval drift (C) — semantic/topic drift
    # ---------------------------------------------------------------
    semantic = check_semantic_drift()
    summary["semantic_drift"] = semantic

    if semantic["semantic_drift"]:
        alerts.append(
            _make_alert(
                "WARNING",
                "Unexpected topic distribution shift detected.",
                semantic,
            )
        )

    # ---------------------------------------------------------------
    # 5. IDK fallback consistency
    # ---------------------------------------------------------------
    idk = check_idk_behavior()
    summary["idk_behavior"] = idk

    if idk["idk_violations"] > 0:
        alerts.append(
            _make_alert(
                "WARNING",
                f"{idk['idk_violations']} fallback answers did NOT use 'I don't know.'.",
                idk,
            )
        )

    # ---------------------------------------------------------------
    # 6. Volume anomalies (throughput)
    # ---------------------------------------------------------------
    volume = check_volume_anomalies()
    summary["volume"] = volume

    if volume["volume_alert"]:
        alerts.append(
            _make_alert(
                "INFO",
                f"High traffic: {volume['rpm']:.2f} requests/min (last hour).",
                volume,
            )
        )

    # ---------------------------------------------------------------
    # Finalize + write results
    # ---------------------------------------------------------------
    _write_outputs(summary, alerts)
    return summary, alerts


# --------------------------------------------------------------------
# Write summary + alerts
# --------------------------------------------------------------------
def _write_outputs(summary, alerts):
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    ALERTS_FILE.write_text(json.dumps(alerts, indent=2))


# --------------------------------------------------------------------
# Manual execution
# --------------------------------------------------------------------
if __name__ == "__main__":
    summary, alerts = run_alert_checks()
    print("\n=== Phase 8 Summary ===")
    print(json.dumps(summary, indent=2))

    print("\n=== Alerts ===")
    for a in alerts:
        print(json.dumps(a, indent=2))
