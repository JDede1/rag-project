"""
main.py
-------------------------------------
FastAPI RAG API:
- Retrieves top-k FAQs from FAISS index
- Generates grounded answers using Phi-3-Mini-4k-Instruct (lazy-loaded)
"""

import sys, os

# ---------------------------------------------------------
# Ensure FastAPI can import from src/ (Colab-compatible)
# ---------------------------------------------------------
# main.py is located at: /content/rag-project/src/api/main.py
# We add the parent folder (/src) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd

# ---------------------------------------------------------
# Correct imports from src/
# ---------------------------------------------------------
from src.retrieval.search_engine import RbcRetriever

# Generator is lazy-loaded inside load_generator()
generate_answer = None
generator_loaded = False

# ---------------------------------------------------------
# App setup
# ---------------------------------------------------------
app = FastAPI(
    title="RBC RAG API",
    description="Retrieval-Augmented Generation service powered by FAISS + Phi-3-Mini-4k-Instruct",
    version="1.0.0",
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
print("🔹 Initializing retriever...")
retriever = RbcRetriever()
print("✅ Retriever ready.\n")

# ---------------------------------------------------------
# Lazy-load the generator on first request
# ---------------------------------------------------------
def load_generator():
    """Load the generator only when first needed."""
    global generate_answer, generator_loaded

    if not generator_loaded:
        print("⚙️ Loading generator model (Phi-3-Mini-4k-Instruct)...")

        # CORRECT import path
        from src.generation.generator import generate_answer as _generate_answer

        generate_answer = _generate_answer
        generator_loaded = True
        print("✅ Generator model loaded.\n")

# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "records": len(retriever.metadata),
        "model": "microsoft/Phi-3-Mini-4k-Instruct",
        "generator_loaded": generator_loaded,
    }

# ---------------------------------------------------------
# ASK Endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User question to search and answer"),
    top_k: int = Query(3, ge=1, le=10, description="Top-k FAQ results to retrieve"),
):
    """
    Step 1: Retrieve FAQs  
    Step 2: Lazy-load generator  
    Step 3: Grounded generation  
    """
    try:
        # 🔍 Step 1 — Retrieve from FAISS
        results_df = retriever.search(query, top_k=top_k)
        retrieved_docs = results_df["answer"].tolist()

        # 🧠 Step 2 — Load generator if needed
        if not generator_loaded:
            load_generator()

        # 🧠 Step 3 — Generate grounded answer
        answer = generate_answer(query, retrieved_docs)

        # 📦 Step 4 — Respond
        return {
            "query": query,
            "answer": answer,
            "context": results_df.to_dict(orient="records"),
        }

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# Run Server
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
