"""
main.py
-------------------------------------------------------
FastAPI RAG backend for RBC banking FAQs.

Stable Version:
    • Uses Qwen2.5-0.5B-Instruct (fast and reliable in Colab)
    • Works perfectly with Cloudflare tunnel
    • Retrieval results handled as list-of-dicts (not DataFrame)
    • Strict grounding and safe JSON outputs
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
from src.generation.generator import generate_answer   # Loaded at startup


# ---------------------------------------------------------
# FastAPI initialization
# ---------------------------------------------------------
app = FastAPI(
    title="RBC RAG API",
    description="Retrieval-Augmented Generation using FAISS + Qwen2.5-0.5B-Instruct",
    version="3.1.0",
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
    Works with Python list-of-dicts from FAISS retrieval.
    """
    if not results:
        return []

    # Sort by similarity score (descending)
    sorted_results = sorted(
        results,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    # Filter based on score cutoff
    filtered = [
        r for r in sorted_results
        if r.get("score", 0) >= score_threshold
    ]

    # Return only chunk text
    chunks = [
        r["chunk"]
        for r in filtered
        if "chunk" in r
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
        "retriever_model": "sentence-transformers/all-mpnet-base-v2",
        "generator_model": "Qwen2.5-0.5B-Instruct",
    }


# ---------------------------------------------------------
# Main RAG endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's question"),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        # Step 1: retrieve from FAISS (returns DataFrame)
        df = retriever.search(query, top_k=top_k)

        # Convert to list-of-dicts for downstream stability
        retrieval_results = df.to_dict(orient="records")

        # Step 2: filter & clean chunks
        cleaned_chunks = clean_retrieval(retrieval_results)

        # Step 3: generate answer
        answer = generate_answer(query, cleaned_chunks)

        # Step 4: return clean JSON
        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieval_results,   # full objects (for debugging)
            "used_context": cleaned_chunks,   # what generator saw
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# Run server manually (if not using uvicorn CLI)
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
