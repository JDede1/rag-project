"""
main.py
-------------------------------------------------------
FastAPI RAG backend for RBC banking FAQs.

Improvements in this version:
    • Compatible with hybrid-grounding generator.py
    • clean_retrieval() now returns chunk text *only for the generator*
    • Full retrieved dicts are still returned to the UI
    • Stable, strict, no hallucinations
"""

import sys
import os

# Allow imports from src/
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
    description="Retrieval-Augmented Generation (FAISS + Hybrid Grounded LLM)",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow Streamlit (Cloudflare)
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
# Clean retrieval results for generator
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.40, max_items: int = 4):
    """
    Returns ONLY the 'chunk' text for the generator.

    UI still receives full 'retrieved' dicts separately.
    """
    if not results:
        return []

    # Sort by score
    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    # Apply score cutoff
    filtered = [r for r in sorted_results if r.get("score", 0) >= score_threshold]

    # Extract only text chunks for the generator
    chunk_texts = [r["chunk"] for r in filtered if isinstance(r.get("chunk"), str)]

    return chunk_texts[:max_items]


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "record_count": len(retriever.metadata),
        "retriever_model": retriever.model_name,
        "generator_model": "Hybrid Grounded — Qwen2.5-0.5B-Instruct",
    }


# ---------------------------------------------------------
# Main Retrieval-Augmented Generation Endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's question"),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        # Step 1 — Retrieve full metadata dicts
        retrieval_results = retriever.search(query, top_k=top_k)

        # Step 2 — Extract clean chunk text for LLM
        cleaned_chunks = clean_retrieval(retrieval_results)

        # Step 3 — Generate grounded answer
        answer = generate_answer(query, cleaned_chunks)

        # Step 4 — Return full metadata + generator answer
        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieval_results,   # full dicts for UI
            "used_context": cleaned_chunks,   # text only for LLM
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# Manual Run
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
