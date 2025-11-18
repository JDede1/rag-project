"""
main.py — FastAPI Backend for RAG (Phase 6 Enhanced)
-------------------------------------------------------
Adds:
    • Citation-aware context cleaning
    • Confidence-based fallback
    • Structured response fields
    • Safe integration with Phase 6 generator + search_engine
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
    description="Retrieval-Augmented Generation using FAISS + Phi-3.5-Mini (Phase 6)",
    version="6.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Streamlit from Cloudflare or ngrok
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
# PHASE 6: Clean Retrieval for Generator
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.32, max_items: int = 4):
    """
    Phase 6 Improvements:
        • Uses final_score instead of raw FAISS score
        • Returns both chunk text AND citation IDs in parallel lists
        • Prevents weak matches from being used
    """
    if not results:
        return [], []

    # sort by reranked score
    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    strong = [
        r for r in ordered
        if r.get("final_score", 0.0) >= score_threshold and isinstance(r.get("chunk"), str)
    ]

    # Generator expects pure chunk strings and their citation IDs
    chunks = [r["chunk"].strip() for r in strong][:max_items]
    citation_ids = [r["citation_id"] for r in strong][:max_items]

    return chunks, citation_ids


# ---------------------------------------------------------
# HTML Landing Page
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse(
        "landing.html",
        {"request": request}
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
        "generator_model": "Structured Grounded — Phi-3.5-Mini-Instruct (Phase 6)"
    }


# ---------------------------------------------------------
# PHASE 6 — Main RAG Endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's banking question"),
    top_k: int = Query(5, ge=1, le=10)
):
    try:
        # Step 1 — retrieval (with reranking + citation IDs)
        retrieval_results = retriever.search(query, top_k=top_k)

        # Step 2 — extract strong context + citation IDs
        clean_chunks, citation_ids = clean_retrieval(retrieval_results)

        # Phase 6 Fallback: if no context is strong enough
        if not clean_chunks:
            return {
                "query": query,
                "answer": "I don't know.",
                "citations_used": [],
                "retrieved": retrieval_results,
                "used_context": [],
                "confidence": 0.0,
            }

        # Step 3 — structured generation
        answer = generate_answer(query, clean_chunks)

        # Step 4 — read confidence from retrieval
        confidence = (
            retrieval_results[0].get("confidence", 0.0)
            if retrieval_results else 0.0
        )

        return {
            "query": query,
            "answer": answer,
            "citations_used": citation_ids,
            "retrieved": retrieval_results,
            "used_context": clean_chunks,
            "confidence": confidence,
        }

    except Exception as e:
        return {
            "query": query,
            "answer": "I don't know.",
            "error": str(e),
            "citations_used": [],
            "used_context": [],
            "retrieved": [],
            "confidence": 0.0,
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
