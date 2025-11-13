"""
main.py
-------------------------------------------------------
FastAPI RAG backend for RBC banking FAQs.

Phase 4 upgrades:
    • Uses chunk-level retriever (mpnet embeddings)
    • Uses Phi-3.5-Mini-Instruct (chat model)
    • Strict grounding: model answers ONLY from retrieved chunks
    • JSON-friendly output structure
    • Cleaner, safer, predictable behavior
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------
# Ensure imports work in Colab / relative repo structure
# ---------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.retrieval.search_engine import RbcRetriever

# Lazy-loaded generator
_generate_answer = None
generator_loaded = False


# ---------------------------------------------------------
# FastAPI initialization
# ---------------------------------------------------------
app = FastAPI(
    title="RBC RAG API",
    description="RBC Banking Retrieval-Augmented Generation using FAISS + Phi-3.5",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Load retriever at startup
# ---------------------------------------------------------
print("Loading retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# Lazy-load generator
# ---------------------------------------------------------
def load_generator():
    global _generate_answer, generator_loaded
    if not generator_loaded:
        print("Loading Phi-3.5 generator...")
        from src.generation.generator import generate_answer as gen
        _generate_answer = gen
        generator_loaded = True
        print("Generator loaded.\n")


# ---------------------------------------------------------
# Retrieval Post-Processing
# ---------------------------------------------------------
def clean_retrieval(results: list, min_score: float = 0.40, max_items: int = 4):
    """
    Filter retrieved chunks before sending to the generator.
    """
    if not results:
        return []

    # Sort descending by similarity
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # Filter based on score threshold
    filtered = [r for r in results if r["score"] >= min_score]

    # Extract only chunk text
    chunks = [r["chunk"] for r in filtered]

    # Restrict count to avoid overloading the prompt
    return chunks[:max_items]


# ---------------------------------------------------------
# Health check endpoint
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "records": len(retriever.metadata),
        "generator_loaded": generator_loaded,
        "retriever_model": "sentence-transformers/all-mpnet-base-v2",
        "generator_model": "microsoft/Phi-3.5-mini-instruct",
    }


# ---------------------------------------------------------
# Main RAG Endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User question"),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        # Step 1: Retrieve top-k semantic matches
        retrieval_results = retriever.search(query, top_k=top_k)

        # Step 2: Clean and filter retrieved chunks
        cleaned_chunks = clean_retrieval(retrieval_results)

        # Step 3: Lazy-load generator
        if not generator_loaded:
            load_generator()

        # Step 4: Generate grounded answer
        answer = _generate_answer(query, cleaned_chunks)

        # Step 5: Return structured response
        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieval_results,       # full objects
            "used_context": cleaned_chunks        # only chunks used
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# Run server if executed directly
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
