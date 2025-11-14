"""
main.py
-------------------------------------------------------
FastAPI RAG backend for RBC banking FAQs.

Upgrades:
    • Uses Qwen2.5-0.5B-Instruct (fast, stable, no timeouts)
    • Loads generator at startup (not lazy-loaded)
    • Uses strict chunk-level grounding
    • Works smoothly with Cloudflare tunnels
    • Clean JSON-safe output
"""

import sys
import os

# ---------------------------------------------------------
# Ensure imports work (Colab + VS Code)
# ---------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd

from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer   # Qwen 0.5B loaded immediately


# ---------------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------------
app = FastAPI(
    title="RBC RAG API",
    description="Retrieval-Augmented Generation using FAISS + Qwen2.5-0.5B-Instruct",
    version="3.0.0",
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
print("Retriever loaded.")


# ---------------------------------------------------------
# Clean & Filter retrieved chunks
# ---------------------------------------------------------
def clean_retrieval(df: pd.DataFrame, score_threshold: float = 0.40, max_items: int = 4):
    """
    Convert FAISS results → clean chunk list for generator.
    """
    if df.empty:
        return []

    # highest scores first
    df = df.sort_values("score", ascending=False)

    # apply score cutoff
    df = df[df["score"] >= score_threshold]

    # extract chunk text
    chunks = df["chunk"].tolist()

    return chunks[:max_items]


# ---------------------------------------------------------
# Health Endpoint
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
# Main RAG Endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User question"),
    top_k: int = Query(5, ge=1, le=10),
):
    try:
        # Step 1: retrieve semantic matches
        retrieval_df = retriever.search(query, top_k=top_k)

        # Step 2: convert for generator
        cleaned_chunks = clean_retrieval(retrieval_df)

        # Step 3: generate grounded answer
        answer = generate_answer(query, cleaned_chunks)

        # Step 4: return structured response
        return {
            "query": query,
            "answer": answer,
            "retrieved": retrieval_df.to_dict(orient="records"),
            "used_context": cleaned_chunks,
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# Run locally
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
