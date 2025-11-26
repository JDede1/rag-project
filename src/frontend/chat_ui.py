"""
chat_ui.py — Streamlit Frontend (Cloudflare Tunnel + Cloud Run)
-----------------------------------------------------------------
Final Architecture (Recommended):
    Streamlit (local)
        ↓ Cloudflare Tunnel (public)
        ↓ Cloud Run FastAPI backend

This file is now:
    • Cloud Run–native (no ngrok, no Colab URLs)
    • Cleaner URL loading
    • Stable structured answer rendering
"""

import streamlit as st
import requests
import os


# ============================================================
# BACKEND URL RESOLUTION
# ============================================================

def load_backend_url():
    """
    Always load backend URL from Streamlit secrets.
    This is the correct long-term source for Cloud Run deployments.
    """
    url = st.secrets.get("RAG_BACKEND_URL")
    return url


BACKEND_URL = load_backend_url()

if not BACKEND_URL:
    st.error("""
    ❌ Backend URL missing!

    Please add this in `.streamlit/secrets.toml`:

    RAG_BACKEND_URL = "https://<your-cloud-run-url>"
    """)
    st.stop()

# Normalize (remove trailing /)
BACKEND_URL = BACKEND_URL.rstrip("/")


# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Fintech AI Assistant",
    page_icon="💬",
    layout="wide"
)


# ============================================================
# LOAD CSS (LOCAL ONLY)
# ============================================================
CSS_PATH = os.path.join("src", "frontend", "static", "style.css")

def load_css(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception:
            pass

load_css(CSS_PATH)


# ============================================================
# HERO BANNER
# ============================================================
st.markdown("""
<div class="hero">
    <h2 style="margin-bottom:4px;">💬 Fintech AI Assistant</h2>
    <p>Ask banking questions. Answers are grounded strictly in the official FAQ knowledge base.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state["messages"] = []      # (role, text, citations)

if "retrieved_docs" not in st.session_state:
    st.session_state["retrieved_docs"] = []

if "waiting" not in st.session_state:
    st.session_state["waiting"] = False


# ============================================================
# RENDER ANSWERS
# ============================================================
def render_answer_block(text: str, citations: list):
    """Renders the sections returned by generator.py (Short Answer, Details, Sources…)."""

    st.markdown(text)

    if citations:
        st.markdown("### 📎 Citations Used")
        cit_text = ", ".join(f"[CIT:{cid}]" for cid in citations)
        st.markdown(f"**{cit_text}**")


# Render chat history
for role, text, citations in st.session_state["messages"]:
    with st.chat_message(role):
        if role == "assistant":
            render_answer_block(text, citations)
        else:
            st.markdown(text)


# ============================================================
# HANDLE USER INPUT
# ============================================================
user_prompt = st.chat_input("Ask any banking-related question...")

if user_prompt and not st.session_state["waiting"]:

    # Add user msg to history
    st.session_state["messages"].append(("user", user_prompt, []))

    with st.chat_message("user"):
        st.markdown(user_prompt)

    placeholder = st.chat_message("assistant")
    placeholder.markdown("<span class='typing'>Thinking...</span>", unsafe_allow_html=True)

    st.session_state["waiting"] = True

    # --------------------------------------------------------
    # Call backend securely
    # --------------------------------------------------------
    try:
        response = requests.get(
            f"{BACKEND_URL}/ask",
            params={"query": user_prompt, "top_k": 5},
            timeout=50,
        )
        data = response.json()

        answer = data.get("answer", "No answer returned.")
        retrieved_docs = data.get("retrieved", [])
        citations_used = data.get("citations_used", [])
        confidence = data.get("confidence", 0.0)

    except Exception as e:
        answer = f"Backend connection failed: {e}"
        retrieved_docs = []
        citations_used = []
        confidence = 0.0

    # Show final answer
    with placeholder:
        render_answer_block(answer, citations_used)
        st.caption(f"🔒 Confidence: {confidence:.3f}")

    # Save history
    st.session_state["messages"].append(("assistant", answer, citations_used))
    st.session_state["retrieved_docs"] = retrieved_docs
    st.session_state["waiting"] = False


# ============================================================
# SIDEBAR — RETRIEVED EVIDENCE
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

        final_score = float(doc.get("final_score", doc.get("score", 0)))
        bar_width = max(5, int(final_score * 100))

        st.sidebar.markdown(
            f"<div class='score-bar' style='width:{bar_width}%;'></div>",
            unsafe_allow_html=True
        )

        st.sidebar.caption(f"Score: {final_score:.3f} | CIT:{doc.get('citation_id', '?')}")

        if doc.get("url"):
            st.sidebar.markdown(f"[Source Link]({doc['url']})")

        st.sidebar.markdown("</div>", unsafe_allow_html=True)
else:
    st.sidebar.info("Ask a question to view retrieved chunks.")


# ============================================================
# DEBUG PANEL (Optional)
# ============================================================
with st.expander("🔍 Raw Retrieval Metadata"):
    st.write(st.session_state["retrieved_docs"])
