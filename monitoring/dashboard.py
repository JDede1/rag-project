"""
dashboard.py — Monitoring Dashboard (Streamlit)
Phase 8 Updated Version
-----------------------------------------------

Adds:
    • Alerts Panel (reads monitoring/alerts.json)
    • Retrieval Drift visualizations
    • Latency percentiles (P50/P95/P99)
    • Topic distribution (semantic drift)
"""

import json
import time
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

from monitoring.load_logs import load_all_logs
from monitoring.analyze_logs import (
    summarize_logs,
    top_queries,
    latency_stats,
    grounding_stats,
    detect_hallucinations,
)

# Phase 8: alert + summary files
ALERTS_FILE = Path("logs/alerts.json")
SUMMARY_FILE = Path("logs/phase8_summary.json")


def main():

    # ---------------------------------------------------------
    # Streamlit Page Config — must be FIRST Streamlit command
    # ---------------------------------------------------------
    st.set_page_config(
        page_title="RAG Monitoring Dashboard",
        layout="wide",
        page_icon="📊"
    )

    st.title("📊 RAG Monitoring Dashboard")
    st.caption("Real-time analytics, drift detection, and alerting for your RAG system.")

    # ---------------------------------------------------------
    # Auto-refresh controls
    # ---------------------------------------------------------
    col1, col2 = st.columns([1, 4])
    with col1:
        autorefresh = st.checkbox("Auto-refresh", value=False)

    with col2:
        refresh_rate = st.slider("Refresh every (seconds)", 2, 30, 5)

    if autorefresh:
        time.sleep(refresh_rate)
        st.rerun()

    # ---------------------------------------------------------
    # Load logs
    # ---------------------------------------------------------
    logs = load_all_logs()
    if not logs:
        st.warning("No RAG logs found yet. Make a query and refresh.")
        return

    df = pd.DataFrame(logs)

    # ---------------------------------------------------------
    # SUMMARY METRICS
    # ---------------------------------------------------------
    st.header("📌 Summary (Phase 8)")

    summary = summarize_logs()
    lat_stats = latency_stats()
    gs_stats = grounding_stats()

    # Percentiles for P50 / P95 / P99
    lat_sorted = sorted(df["latency_ms"].tolist())
    p50 = lat_sorted[int(0.50 * (len(lat_sorted) - 1))]
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
    p99 = lat_sorted[int(0.99 * (len(lat_sorted) - 1))]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Requests", summary.get("total_logs", 0))
    c2.metric("Avg Latency (ms)", round(summary.get("avg_latency_ms", 0), 2))
    c3.metric("Avg Grounding Score", summary.get("avg_grounding_score", 0))
    c4.metric("P95 Latency (ms)", round(p95, 2))
    c5.metric("P99 Latency (ms)", round(p99, 2))

    # ---------------------------------------------------------
    # PHASE 8 ALERT PANEL
    # ---------------------------------------------------------
    st.header("🚨 Alerts (Phase 8)")

    if ALERTS_FILE.exists():
        alerts = json.loads(ALERTS_FILE.read_text())

        if not alerts:
            st.success("No active alerts — system is healthy.")
        else:
            for alert in alerts:
                sev = alert["severity"]
                msg = alert["message"]
                data = alert.get("data", {})

                if sev == "CRITICAL":
                    st.error(f"🔴 {msg}")
                elif sev == "WARNING":
                    st.warning(f"🟠 {msg}")
                else:
                    st.info(f"🔵 {msg}")

                with st.expander("Details"):
                    st.json(data)

    else:
        st.info("No alerts.json file found yet. Run Phase 8 alert engine.")

    # ---------------------------------------------------------
    # Latency Chart
    # ---------------------------------------------------------
    st.subheader("⏱ Latency Over Time")

    df_sorted = df.sort_values("timestamp")
    fig_latency = px.line(
        df_sorted,
        x="timestamp",
        y="latency_ms",
        title="Latency (ms) over time",
        markers=True
    )
    st.plotly_chart(fig_latency, use_container_width=True)

    # ---------------------------------------------------------
    # Grounding Score Chart
    # ---------------------------------------------------------
    st.subheader("📚 Grounding Score Over Time")

    fig_grounding = px.line(
        df_sorted,
        x="timestamp",
        y="grounding_score",
        title="Grounding Score over time",
        markers=True,
        range_y=[0, 1]
    )
    st.plotly_chart(fig_grounding, use_container_width=True)

    # ---------------------------------------------------------
    # Retrieval Confidence Drift (Phase 8)
    # ---------------------------------------------------------
    st.subheader("📉 Retrieval Confidence Over Time")

    fig_conf = px.line(
        df_sorted,
        x="timestamp",
        y="confidence",
        title="Retriever Confidence Drift",
        markers=True,
        range_y=[0, 1]
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # ---------------------------------------------------------
    # Semantic Drift — Topic Distribution
    # ---------------------------------------------------------
    st.subheader("🧠 Semantic Drift — Topic Distribution (Top Retrieved Chunk)")

    topics = []
    for r in logs:
        retrieved = r.get("retrieved", [])
        if retrieved:
            topics.append(retrieved[0].get("topic", "unknown"))
        else:
            topics.append("unknown")

    topic_df = pd.DataFrame({"topic": topics})

    fig_topics = px.histogram(
        topic_df,
        x="topic",
        title="Distribution of Top Retrieved Topics"
    )
    st.plotly_chart(fig_topics, use_container_width=True)

    # ---------------------------------------------------------
    # Query Frequency
    # ---------------------------------------------------------
    st.subheader("💬 Most Frequent Queries")

    top_q = top_queries(10)
    freq_df = pd.DataFrame(top_q, columns=["query", "count"])

    fig_freq = px.bar(
        freq_df,
        x="query",
        y="count",
        title="Top 10 User Queries",
        text="count"
    )
    fig_freq.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_freq, use_container_width=True)

    # ---------------------------------------------------------
    # Hallucination Detection
    # ---------------------------------------------------------
    st.subheader("⚠️ Potential Hallucinations")

    hall = detect_hallucinations(threshold=0.50)

    if not hall:
        st.success("No hallucinations detected!")
    else:
        hall_df = pd.DataFrame(hall)
        st.error(f"{len(hall)} potential hallucinations found.")
        st.dataframe(hall_df[[ 
            "timestamp", "query", "answer", "grounding_score", "context_overlap"
        ]])

    # ---------------------------------------------------------
    # Raw Log Viewer
    # ---------------------------------------------------------
    st.subheader("📄 Raw Log Table")
    st.dataframe(df, height=400)

    # ---------------------------------------------------------
    # Download Logs
    # ---------------------------------------------------------
    st.subheader("⬇️ Download Logs")

    log_file = Path("logs/rag_requests.jsonl")

    if log_file.exists():
        st.download_button(
            label="Download raw JSONL logs",
            data=log_file.read_text(),
            file_name="rag_requests.jsonl",
            mime="application/jsonl"
        )
    else:
        st.warning("Log file not found.")

    st.success("Dashboard loaded successfully.")


# ---------------------------------------------------------
# SAFETY GUARD — Prevent double execution in Colab/Streamlit
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
