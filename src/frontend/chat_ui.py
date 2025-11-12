"""
chat_ui.py
-------------------------------------
Streamlit Chat UI for RAG system with evidence viewer.
"""

import streamlit as st
import requests
import os
import pandas as pd

# ✅ Backend URL (set automatically from file or env)
URL_FILE = "/content/rag-project/rag_llm_url.txt"
if os.path.exists(URL_FILE):
    BACKEND_URL = open(URL_FILE).read().strip()
else:
    BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="💬 RBC RAG Assistant", layout="wide")

# --- Header ---
st.title("💬 RBC AI Assistant")
st.caption("Ask questions about RBC banking FAQs")

# --- Sidebar ---
st.sidebar.header("📚 Retrieved FAQ Evidence")
st.sidebar.info("Relevant FAQs will appear here after you ask a question.")

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "context_docs" not in st.session_state:
    st.session_state["context_docs"] = []

# --- Display chat history ---
for role, text in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(text)

# --- User Input ---
if prompt := st.chat_input("Type your question here..."):
    st.session_state["messages"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Call backend ---
    try:
        resp = requests.get(f"{BACKEND_URL}/ask", params={"query": prompt, "top_k": 3})
        data = resp.json()
        answer = data.get("answer", "⚠️ No answer received.")
        context = data.get("context", [])
    except Exception as e:
        answer = f"⚠️ Connection error: {e}"
        context = []

    # --- Display model response ---
    with st.chat_message("assistant"):
        st.markdown(answer)

    # --- Save messages & evidence ---
    st.session_state["messages"].append(("assistant", answer))
    st.session_state["context_docs"] = context

# --- Sidebar Evidence Viewer ---
if st.session_state["context_docs"]:
    for i, doc in enumerate(st.session_state["context_docs"], start=1):
        st.sidebar.markdown(f"**{i}. {doc['question']}**")
        st.sidebar.write(doc["answer"][:400] + ("..." if len(doc["answer"]) > 400 else ""))
        st.sidebar.divider()
