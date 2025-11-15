"""
main.py
-------------------------------------------------------
FastAPI backend for the RBC RAG system.

Aligned with:
    • Phi-3.5-Mini-Instruct generator (strict no-hallucination mode)
    • Hybrid-grounding pipeline
    • Clean chunk extraction
    • Full metadata passthrough for Streamlit UI
"""

import sys
import os

# Ensure src/ is importable
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
    description="Retrieval-Augmented Generation using FAISS + Phi-3.5-Mini-Instruct",
    version="5.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Needed for Streamlit via Cloudflare
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Load Retriever Once
# ---------------------------------------------------------
print("Loading retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# Chunk Cleaning for Generator Input
# ---------------------------------------------------------
def clean_retrieval(
    results: list,
    score_threshold: float = 0.40,
    max_items: int = 4
):
    """
    Input to Phi-3.5 MUST be ONLY raw text chunks.
    Never pass dicts or metadata to the generator.
    """

    if not results:
        return []

    # Sort descending by similarity score
    sorted_results = sorted(
        results,
        key=lambda r: r.get("score", 0.0),
        reverse=True
    )

    # Apply score threshold
    filtered = [
        r for r in sorted_results
        if r.get("score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    # Extract raw text only
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
        "generator_model": "Hybrid Grounded — Phi-3.5-Mini-Instruct"
    }


# ---------------------------------------------------------
# Main RAG Endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's question for the RAG system"),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        # Step 1 — Retrieve full metadata dicts
        retrieval_results = retriever.search(query, top_k=top_k)

        # Step 2 — Extract pure chunk text for LLM
        clean_chunks = clean_retrieval(retrieval_results)

        # Step 3 — Generate answer using hybrid-grounded Phi model
        answer = generate_answer(query, clean_chunks)

        # Safety fallback
        if not isinstance(answer, str) or answer.strip() == "":
            answer = "I don't know."

        # Step 4 — Return everything (UI depends on this structure)
        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieval_results,   # full metadata dicts
            "used_context": clean_chunks,     # pure text for LLM
        }

    except Exception as e:
        # Never crash the API — always respond safely
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
