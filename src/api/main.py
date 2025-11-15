"""
main.py
-------------------------------------------------------
FastAPI backend for the RBC RAG system.

Updated for:
    • Phi-3.5-Mini-Instruct generator
    • Hybrid-grounding pipeline
    • Clean chunk extraction
    • Full metadata passthrough for UI
"""

import sys
import os

# Make sure imports work from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer


# ---------------------------------------------------------
# FastAPI Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="RBC RAG API",
    description="Retrieval-Augmented Generation (FAISS + Phi-3.5-Mini + Hybrid Grounding)",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # OK for Streamlit (Cloudflare)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Load Retriever Once at Startup
# ---------------------------------------------------------
print("Loading retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# Extract only the minimal chunk text needed by generator
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.40, max_items: int = 4):
    """
    Generator must receive ONLY raw chunk text—not metadata,
    not questions, not dicts.

    The Streamlit UI will still receive full 'retrieved' dicts.
    """
    if not results:
        return []

    # Sort by similarity score (descending)
    results_sorted = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)

    # Apply quality cutoff
    filtered = [
        r for r in results_sorted
        if r.get("score", 0.0) >= score_threshold and isinstance(r.get("chunk"), str)
    ]

    # Extract pure text chunks
    chunks = [r["chunk"].strip() for r in filtered if r.get("chunk")]

    return chunks[:max_items]


# ---------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "record_count": len(retriever.metadata),
        "retriever_model": retriever.model_name,
        "generator_model": "Hybrid Grounded — Phi-3.5-Mini-Instruct",
    }


# ---------------------------------------------------------
# Main RAG Endpoint (/ask)
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's question"),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        # Step 1 — Retrieve top-k FAISS results
        retrieval_results = retriever.search(query, top_k=top_k)

        # Step 2 — Extract clean text chunks for the LLM
        clean_chunks = clean_retrieval(retrieval_results)

        # Step 3 — Generate grounded answer
        answer = generate_answer(query, clean_chunks)

        # Failsafe: If generator crashes or yields null
        if not isinstance(answer, str) or answer.strip() == "":
            answer = "I don't know."

        # Step 4 — Return everything
        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieval_results,    # full dicts for UI
            "used_context": clean_chunks,      # only text for LLM
        }

    except Exception as e:
        return {
            "query": query,
            "answer": "I don't know.",
            "error": str(e)
        }


# ---------------------------------------------------------
# Manual Local Run
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
