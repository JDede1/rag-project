"""
chat_ui.py — Phase 6 Enhanced UI for RAG System
------------------------------------------------------------
Adds:
    • Structured answer rendering
    • Citations panel
    • Confidence display
    • Clean formatting for multi-section answers
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
# Load External CSS Only
# ============================================================
CSS_PATH = "src/frontend/static/style.css"

def load_css(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except:
            pass  # Silent fail

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
    st.session_state["messages"] = []  # (role, text, citations)

if "retrieved_docs" not in st.session_state:
    st.session_state["retrieved_docs"] = []

if "waiting" not in st.session_state:
    st.session_state["waiting"] = False


# ============================================================
# Render Chat History (Phase 6 formatting)
# ============================================================
def render_answer_block(text: str, citations: list):
    """
    Render the structured answer returned by Phase 6 generator.py.
    """

    st.markdown(text)

    if citations:
        st.markdown("### 📎 Citations Used")
        cit_text = ", ".join(f"[CIT:{cid}]" for cid in citations)
        st.markdown(f"**{cit_text}**")


for role, text, citations in st.session_state["messages"]:
    with st.chat_message(role):
        if role == "assistant":
            render_answer_block(text, citations)
        else:
            st.markdown(text)


# ============================================================
# Handle User Input
# ============================================================
user_prompt = st.chat_input("Ask any banking-related question...")

if user_prompt and not st.session_state["waiting"]:

    st.session_state["messages"].append(("user", user_prompt, []))

    with st.chat_message("user"):
        st.markdown(user_prompt)

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
        citations_used = data.get("citations_used", [])
        confidence = data.get("confidence", 0.0)

    except Exception as e:
        answer = f"Backend connection error: {e}"
        retrieved_docs = []
        citations_used = []
        confidence = 0.0

    # Replace placeholder with structured answer
    with placeholder:
        render_answer_block(answer, citations_used)
        st.caption(f"🔒 Confidence: {confidence:.3f}")

    # Update session state
    st.session_state["messages"].append(("assistant", answer, citations_used))
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
# Debug Panel
# ============================================================
with st.expander("🔍 Show Raw Retrieval Metadata (Debug Mode)"):
    st.write(st.session_state["retrieved_docs"])
