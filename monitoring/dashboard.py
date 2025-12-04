"""
dashboard.py — Monitoring Dashboard (Streamlit)
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
    st.caption("Real-time analytics and quality monitoring for your RAG system.")

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
    # Summary Metrics
    # ---------------------------------------------------------
    summary = summarize_logs()
    lat_stats = latency_stats()
    gs_stats = grounding_stats()

    st.header("📌 Summary")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Requests", summary.get("total_logs", 0))
    c2.metric("Avg Latency (ms)", round(summary.get("avg_latency_ms", 0), 2))
    c3.metric("Avg Grounding Score", summary.get("avg_grounding_score", 0))
    c4.metric("Median Latency (ms)", round(lat_stats.get("median_ms", 0), 2))

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
