"""
chat_ui.py — Premium Fintech UI for RAG System
------------------------------------------------------------
Safe, non-RBC-branded. Modern banking aesthetics.
Works with:
    - FastAPI backend (ngrok)
    - Streamlit Frontend (Cloudflare)
    - Phi-3.5-Mini + Hybrid-grounded generator
"""

import streamlit as st
import requests
import os
import time


# ============================================================
# Resolve Backend URL
# ============================================================
COLAB_URL_FILE = "/content/rag-project/rag_llm_url.txt"

if os.path.exists(COLAB_URL_FILE):
    BACKEND_URL = open(COLAB_URL_FILE).read().strip()
else:
    BACKEND_URL = st.secrets.get("RAG_BACKEND_URL")

if not BACKEND_URL:
    st.error("Backend URL missing. Cannot start UI.")
    st.stop()


# ============================================================
# Page Setup
# ============================================================
st.set_page_config(
    page_title="Fintech AI Assistant",
    page_icon="💬",
    layout="wide"
)


# ============================================================
# Custom CSS (Glass / Fintech Aesthetic)
# ============================================================
st.markdown("""
<style>

body {
    background-color: #F4F6F9;
}

/* Hero Banner */
.hero {
    padding: 24px;
    background: linear-gradient(135deg, #0A1A2F, #1C3453);
    border-radius: 18px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.1);
    color: white;
}

/* Chat bubbles (Glassmorphism) */
.stChatMessage {
    border-radius: 14px !important;
    padding: 14px !important;
    margin-bottom: 12px !important;
    backdrop-filter: blur(12px);
}

.stChatMessage.user {
    background: rgba(180, 210, 255, 0.55) !important;
    border: 1px solid rgba(120, 160, 255, 0.35) !important;
    color: #0A1A2F !important;
}

.stChatMessage.assistant {
    background: rgba(255, 255, 255, 0.70) !important;
    border: 1px solid rgba(200,200,200,0.45) !important;
}

/* Sidebar Cards */
.sidebar-card {
    padding: 14px;
    background: white;
    border-radius: 14px;
    border: 1px solid #E2E6EA;
    margin-bottom: 18px;
}

.score-bar {
    height: 7px;
    border-radius: 4px;
    background: linear-gradient(90deg, #1E90FF, #87CEFA);
}

/* Typing loader */
.typing {
    font-style: italic;
    opacity: 0.7;
}

header, footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# ============================================================
# Hero Banner
# ============================================================
st.markdown("""
<div class="hero">
    <h2 style="margin-bottom:4px;">💬 Fintech AI Assistant</h2>
    <p>Ask banking questions. Answers are grounded strictly in your scraped FAQ knowledge base.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "retrieved_docs" not in st.session_state:
    st.session_state["retrieved_docs"] = []

if "loading" not in st.session_state:
    st.session_state["loading"] = False


# ============================================================
# Render Chat History
# ============================================================
for role, text in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(text)


# ============================================================
# User Input
# ============================================================
user_prompt = st.chat_input("Ask any banking-related question...")

if user_prompt:
    st.session_state["messages"].append(("user", user_prompt))

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Backend call
    st.session_state["loading"] = True
    with st.chat_message("assistant"):
        st.markdown("<span class='typing'>Thinking...</span>", unsafe_allow_html=True)

    try:
        response = requests.get(
            f"{BACKEND_URL}/ask",
            params={"query": user_prompt, "top_k": 5},
            timeout=60
        )
        data = response.json()

        answer = data.get("answer", "No answer returned.")
        retrieved_docs = data.get("retrieved", [])

    except Exception as e:
        answer = f"Backend connection error: {e}"
        retrieved_docs = []

    # Remove loader, replace with final answer
    st.session_state["messages"].append(("assistant", answer))
    st.session_state["retrieved_docs"] = retrieved_docs
    st.session_state["loading"] = False

    st.rerun()


# ============================================================
# Sidebar — Evidence Viewer
# ============================================================
st.sidebar.header("Retrieved Evidence", divider="gray")

retrieved = st.session_state["retrieved_docs"]

if retrieved:
    for i, doc in enumerate(retrieved, start=1):
        with st.sidebar:
            st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)

            st.markdown(f"**{i}. {doc.get('question','(No question)')}**")

            st.markdown(
                f"<div class='sidebar-chunk'>{doc.get('chunk','')[:350]}...</div>",
                unsafe_allow_html=True
            )

            # Score bar
            score = float(doc.get("score", 0))
            bar_width = int(score * 100)

            st.markdown(
                f"<div class='score-bar' style='width:{bar_width}%;'></div>",
                unsafe_allow_html=True
            )

            st.caption(f"Match Score: {score:.3f}")

            if doc.get("url"):
                st.markdown(f"[Source Link]({doc['url']})")

            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.sidebar.info("Ask a question to view retrieved chunks.")


# ============================================================
# Optional Developer Panel
# ============================================================
with st.expander("🔍 Show Raw Retrieval Metadata (Debug Mode)"):
    st.write(st.session_state["retrieved_docs"])
