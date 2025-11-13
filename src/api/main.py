"""
main.py
-------------------------------------
FastAPI RAG API:
- Retrieves top-k FAQs from FAISS index
- Generates grounded answers using Phi-3-Mini-4k-Instruct (lazy-loaded)
- Includes strong anti-hallucination filtering
"""

import sys, os, re

# ---------------------------------------------------------
# Ensure FastAPI can import from src/ (Colab-compatible)
# ---------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd

# Correct import
from src.retrieval.search_engine import RbcRetriever

# Generator is lazy-loaded
generate_answer = None
generator_loaded = False

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI(
    title="RBC RAG API",
    description="Retrieval-Augmented Generation (RAG) with FAISS + Phi-3-Mini-4k-Instruct",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Load Retriever at startup
# ---------------------------------------------------------
print("Initializing retriever...")
retriever = RbcRetriever()
print("Retriever ready.\n")

# ---------------------------------------------------------
# Load Generator Lazily
# ---------------------------------------------------------
def load_generator():
    """Load the Phi-3 generator only on first request."""
    global generate_answer, generator_loaded

    if not generator_loaded:
        print("Loading generator model (Phi-3-Mini-4k-Instruct)...")
        from src.generation.generator import generate_answer as _generate_answer

        generate_answer = _generate_answer
        generator_loaded = True
        print("Generator model loaded.\n")

# ---------------------------------------------------------
# Helper — Clean retrieved FAQ context (anti-hallucination)
# ---------------------------------------------------------
def clean_context(text: str) -> str:
    """Clean noisy FAQ text so model cannot hallucinate."""
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove very long strings (marketing blocks)
    if len(text) > 500:
        text = text[:500]

    return text


def filter_retrieval(df: pd.DataFrame, score_threshold=0.45):
    """Filter FAISS results to prevent irrelevant context."""
    if df.empty:
        return []

    # Sort by similarity score descending
    df = df.sort_values("score", ascending=False)

    # Apply threshold
    df = df[df["score"] >= score_threshold]

    # Clean each answer
    cleaned = []
    for _, row in df.iterrows():
        cleaned_text = clean_context(row["answer"])
        if cleaned_text:
            cleaned.append(cleaned_text)

    return cleaned[:3]  # keep max 3 clean items


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "records": len(retriever.metadata),
        "generator_loaded": generator_loaded,
        "model": "microsoft/Phi-3-Mini-4k-Instruct",
    }

# ---------------------------------------------------------
# ASK Endpoint (Hallucination-Safe RAG)
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User question"),
    top_k: int = Query(3, ge=1, le=10),
):
    try:
        # Step 1 — Retrieve
        results_df = retriever.search(query, top_k=top_k)

        # Step 2 — Clean + filter retrieved documents
        cleaned_docs = filter_retrieval(results_df)

        # Step 3 — Lazy-load generator
        if not generator_loaded:
            load_generator()

        # Step 4 — Generate grounded answer
        answer = generate_answer(query, cleaned_docs)

        # Step 5 — Response format
        return {
            "query": query,
            "answer": answer,
            "context": results_df.to_dict(orient="records"),
            "cleaned_used_context": cleaned_docs,  # DEBUG: remove later if you want
        }

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# Run Server Locally
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
