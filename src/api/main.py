"""
main.py
-------------------------------------------------------
FastAPI RAG backend for RBC banking FAQs.

Stable Version (Corrected):
    • Uses Qwen2.5-0.5B-Instruct (fast + reliable in Colab)
    • Retrieval returns list-of-dicts (correct handling)
    • Strict grounding (no hallucinations)
    • Fully JSON-safe
"""

import sys
import os

# ---------------------------------------------------------
# Ensure imports resolve from src/
# ---------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer   # Loaded immediately


# ---------------------------------------------------------
# FastAPI initialization
# ---------------------------------------------------------
app = FastAPI(
    title="RBC RAG API",
    description="Retrieval-Augmented Generation using FAISS + Qwen2.5-0.5B-Instruct",
    version="3.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Load retriever once at startup
# ---------------------------------------------------------
print("Loading retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# Clean & Filter retrieval results (list-of-dicts)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.40, max_items: int = 4):
    """
    Works with list-of-dicts returned by retriever.search().
    """
    if not results:
        return []

    # Highest scores first
    sorted_results = sorted(
        results,
        key=lambda x: x.get("score", 0.0),
        reverse=True,
    )

    # Apply score cutoff
    filtered = [
        r for r in sorted_results
        if r.get("score", 0.0) >= score_threshold
    ]

    # Extract chunk text only
    chunks = [
        r["chunk"]
        for r in filtered
        if "chunk" in r and isinstance(r["chunk"], str)
    ]

    return chunks[:max_items]


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "records": len(retriever.metadata),
        "retriever_model": retriever.model_name,
        "generator_model": "Qwen2.5-0.5B-Instruct",
    }


# ---------------------------------------------------------
# Main RAG endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's banking question"),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        # Step 1 — Retrieve (returns LIST, not DF)
        retrieval_results = retriever.search(query, top_k=top_k)

        # Step 2 — Clean chunks for generator
        cleaned_chunks = clean_retrieval(retrieval_results)

        # Step 3 — Generate grounded answer
        answer = generate_answer(query, cleaned_chunks)

        # Step 4 — JSON response (no DataFrame conversion)
        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieval_results,   # full info
            "used_context": cleaned_chunks,
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# Run server manually
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
