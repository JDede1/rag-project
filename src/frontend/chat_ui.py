"""
chat_ui.py — Premium Fintech UI
-------------------------------------
Streamlit Chat UI for the RAG System.
Modern banking-style design (safe, non-RBC branding).
Works with:
    - FastAPI backend
    - Hybrid-grounding generator
    - FAISS retrieval returning full dicts
"""

import streamlit as st
import requests
import os

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
st.set_page_config(page_title="Banking AI Assistant", layout="wide")

# ============================================================
# Custom Premium CSS (Safe Fintech Aesthetic)
# ============================================================
st.markdown("""
<style>

body {
    background-color: #F5F7FA;
}

/* Hero Banner */
.hero {
    padding: 25px;
    background: linear-gradient(90deg, #0A1A2F, #1E3A5F);
    color: white;
    border-radius: 12px;
    margin-bottom: 25px;
    border: 1px solid #0A1A2F;
}

/* Chat bubbles */
.stChatMessage.user {
    background: #E6F0FF !important;
    border: 1px solid #C2D7FF !important;
    border-radius: 12px !important;
    padding: 12px !important;
    color: #0A1A2F !important;
}

.stChatMessage.assistant {
    background: #FFFFFF !important;
    border: 1px solid #E5E8EB !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Sidebar styling */
.sidebar-card {
    padding: 12px;
    background: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #E2E6EA;
    margin-bottom: 20px;
}

.sidebar-title {
    font-size: 18px;
    font-weight: 600;
}

.sidebar-chunk {
    font-size: 14px;
    color: #3A4A5A;
}

/* Improve chat input box */
textarea, .stTextInput input {
    border-radius: 10px !important;
}

/* Remove Streamlit default decoration */
header, footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Hero Banner
# ============================================================
st.markdown("""
<div class="hero">
    <h2>💬 Banking AI Assistant</h2>
    <p>Your intelligent FAQ assistant powered by retrieval-augmented generation.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "retrieved_docs" not in st.session_state:
    st.session_state["retrieved_docs"] = []

# ============================================================
# Render Chat History
# ============================================================
for role, text in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(text)

# ============================================================
# Handle User Input
# ============================================================
user_prompt = st.chat_input("Ask a banking question...")

if user_prompt:
    # Store user message
    st.session_state["messages"].append(("user", user_prompt))

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Backend call
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

    # Assistant reply
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state["messages"].append(("assistant", answer))
    st.session_state["retrieved_docs"] = retrieved_docs

# ============================================================
# Sidebar — Evidence Viewer
# ============================================================
st.sidebar.markdown("<h3 class='sidebar-title'>📚 Retrieved Evidence</h3>", unsafe_allow_html=True)

retrieved = st.session_state["retrieved_docs"]

if retrieved:
    for i, doc in enumerate(retrieved, start=1):
        with st.sidebar:
            st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)

            st.markdown(f"**{i}. {doc.get('question', '(No question)')}**")
            st.markdown(f"<div class='sidebar-chunk'>{doc.get('chunk','')[:350]}...</div>", unsafe_allow_html=True)
            st.caption(f"Score: {doc.get('score', 0):.3f}")

            if doc.get("url"):
                st.markdown(f"[Source Link]({doc['url']})")

            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.sidebar.info("Ask a question to view retrieved chunks.")
