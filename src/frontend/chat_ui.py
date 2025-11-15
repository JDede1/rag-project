"""
chat_ui.py 
-------------------------------------
Streamlit Chat UI for RAG system with evidence viewer.
"""

import streamlit as st
import requests
import os

# ============================================================
# Backend URL
# ============================================================
COLAB_URL_FILE = "/content/rag-project/rag_llm_url.txt"

if os.path.exists(COLAB_URL_FILE):
    BACKEND_URL = open(COLAB_URL_FILE).read().strip()
else:
    BACKEND_URL = st.secrets.get("RAG_BACKEND_URL")

if not BACKEND_URL:
    st.error("Backend URL missing.")
    st.stop()

# ============================================================
# UI Setup
# ============================================================
st.set_page_config(page_title="💬 RBC RAG Assistant", layout="wide")

st.title("💬 RBC AI Assistant")
st.caption("Ask questions about RBC banking FAQs.")

# Session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "retrieved_docs" not in st.session_state:
    st.session_state["retrieved_docs"] = []

# Render chat history
for role, text in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(text)

# ============================================================
# Handle User Input
# ============================================================
if prompt := st.chat_input("Type your banking question here..."):
    st.session_state["messages"].append(("user", prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        resp = requests.get(
            f"{BACKEND_URL}/ask",
            params={"query": prompt, "top_k": 5},
            timeout=45
        )
        data = resp.json()

        answer = data.get("answer", "No answer received.")
        retrieved = data.get("retrieved", [])   # <-- FIXED

    except Exception as e:
        answer = f"Connection error: {e}"
        retrieved = []

    # Show model answer
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state["messages"].append(("assistant", answer))
    st.session_state["retrieved_docs"] = retrieved

# ============================================================
# Sidebar Evidence Viewer (FIXED)
# ============================================================
st.sidebar.header("Retrieved Evidence")

docs = st.session_state["retrieved_docs"]

if docs:
    for i, doc in enumerate(docs, start=1):
        st.sidebar.markdown(f"### {i}. {doc.get('question','(No question)')}")
        st.sidebar.write(doc.get("chunk", "")[:400] + "...")
        st.sidebar.caption(f"Score: {doc.get('score',0):.3f}")
        if doc.get("url"):
            st.sidebar.write(f"[Source]({doc['url']})")
        st.sidebar.divider()
else:
    st.sidebar.info("Ask a question to see retrieved context.")
