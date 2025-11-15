"""
main.py — FastAPI Backend for RAG
-------------------------------------------------------
Adds:
    • Beautiful Fintech Glass Landing Page (/, /landing)
    • Jinja2 template support
    • Static file support for CSS (optional)
    • Keeps all existing RAG logic unchanged
"""

import sys
import os
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Allow src/ imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer


# ---------------------------------------------------------
# FastAPI Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Fintech RAG API",
    description="Retrieval-Augmented Generation using FAISS + Phi-3.5-Mini",
    version="6.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Allow Streamlit from Cloudflare
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Static & Template Setup
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Only mount static if exists (prevents errors)
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ---------------------------------------------------------
# Load Retriever Once
# ---------------------------------------------------------
print("Loading retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# Clean retrieval for generator
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.40, max_items: int = 4):
    if not results:
        return []

    sorted_results = sorted(
        results,
        key=lambda r: r.get("score", 0.0),
        reverse=True
    )

    filtered = [
        r for r in sorted_results
        if r.get("score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    chunks = [r["chunk"].strip() for r in filtered]
    return chunks[:max_items]


# ---------------------------------------------------------
# HTML Landing Page (Fintech Glass UI)
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse(
        "landing.html",
        {
            "request": request
        }
    )


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "record_count": len(retriever.metadata),
        "retriever_model": retriever.model_name,
        "generator_model": "Hybrid Grounded — Phi-3.5-Mini-Instruct"
    }


# ---------------------------------------------------------
# Main RAG Endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's banking question"),
    top_k: int = Query(5, ge=1, le=10)
):
    try:
        retrieval_results = retriever.search(query, top_k=top_k)
        clean_chunks = clean_retrieval(retrieval_results)
        answer = generate_answer(query, clean_chunks)

        if not answer or not isinstance(answer, str):
            answer = "I don't know."

        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieval_results,
            "used_context": clean_chunks,
        }

    except Exception as e:
        return {
            "query": query,
            "answer": "I don't know.",
            "error": str(e)
        }


# ---------------------------------------------------------
# Manual Run
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
