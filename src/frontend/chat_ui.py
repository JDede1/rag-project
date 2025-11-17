"""
chat_ui.py — Premium Fintech UI for RAG System
------------------------------------------------------------
Clean version with:
    - External style.css only (no inline CSS conflicts)
    - Modern fintech UI
    - Full backend compatibility with FastAPI/ngrok
    - Stable Streamlit chat rendering
"""

import streamlit as st
import requests
import os


# ============================================================
# Backend URL Resolver
# ============================================================
COLAB_URL_FILE = "/content/rag-project/rag_llm_url.txt"

def load_backend_url():
    """Load backend URL from local file or Streamlit secrets."""
    if os.path.exists(COLAB_URL_FILE):
        try:
            url = open(COLAB_URL_FILE).read().strip()
            if url:
                return url
        except:
            pass
    return st.secrets.get("RAG_BACKEND_URL")


BACKEND_URL = load_backend_url()

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
# Load External CSS Only (Final Version)
# ============================================================
CSS_PATH = "src/frontend/static/style.css"

def load_css(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except:
            pass  # Silent fail for safety

load_css(CSS_PATH)


# ============================================================
# Hero Banner
# ============================================================
st.markdown("""
<div class="hero">
    <h2 style="margin-bottom:4px;">💬 Fintech AI Assistant</h2>
    <p>Ask banking questions. Answers are grounded strictly in your FAQ knowledge base.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Session State Setup
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "retrieved_docs" not in st.session_state:
    st.session_state["retrieved_docs"] = []

if "waiting" not in st.session_state:
    st.session_state["waiting"] = False


# ============================================================
# Render Chat History
# ============================================================
for role, text in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(text)


# ============================================================
# Handle User Input
# ============================================================
user_prompt = st.chat_input("Ask any banking-related question...")

if user_prompt and not st.session_state["waiting"]:

    # Add user message
    st.session_state["messages"].append(("user", user_prompt))

    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Placeholder for assistant typing
    placeholder = st.chat_message("assistant")
    placeholder.markdown("<span class='typing'>Thinking...</span>", unsafe_allow_html=True)

    st.session_state["waiting"] = True

    # Call backend
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

    # Replace placeholder
    placeholder.markdown(answer)

    # Update state
    st.session_state["messages"].append(("assistant", answer))
    st.session_state["retrieved_docs"] = retrieved_docs
    st.session_state["waiting"] = False


# ============================================================
# Sidebar — Retrieved Evidence
# ============================================================
st.sidebar.header("Retrieved Evidence", divider="gray")

retrieved = st.session_state["retrieved_docs"]

if retrieved:
    for i, doc in enumerate(retrieved, start=1):

        st.sidebar.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)

        st.sidebar.markdown(f"**{i}. {doc.get('question','(No question)')}**")

        st.sidebar.markdown(
            f"<div class='sidebar-chunk'>{doc.get('chunk','')[:350]}...</div>",
            unsafe_allow_html=True
        )

        score = float(doc.get("score", 0))
        bar_width = max(5, int(score * 100))

        st.sidebar.markdown(
            f"<div class='score-bar' style='width:{bar_width}%;'></div>",
            unsafe_allow_html=True
        )

        st.sidebar.caption(f"Match Score: {score:.3f}")

        if doc.get("url"):
            st.sidebar.markdown(f"[Source Link]({doc['url']})")

        st.sidebar.markdown("</div>", unsafe_allow_html=True)

else:
    st.sidebar.info("Ask a question to view retrieved chunks.")


# ============================================================
# Debug Panel
# ============================================================
with st.expander("🔍 Show Raw Retrieval Metadata (Debug Mode)"):
    st.write(st.session_state["retrieved_docs"])
