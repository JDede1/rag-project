"""
main.py
-------------------------------------
FastAPI RAG API:
- Retrieves top-k FAQs from FAISS index
- Generates grounded answers using LLaMA-3.1-8B-Instruct
"""

import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from retrieval.search_engine import RbcRetriever
from generation.generator import generate_answer
import uvicorn
import pandas as pd

# ---------------------------------------------------------
# App setup
# ---------------------------------------------------------
app = FastAPI(
    title="RBC RAG API",
    description="Retrieval-Augmented Generation service powered by FAISS + LLaMA-3.1-8B",
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
# Load Retriever
# ---------------------------------------------------------
print("🔹 Initializing retriever...")
retriever = RbcRetriever()
print("✅ Retriever ready.\n")

# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "records": len(retriever.metadata),
        "model": "meta-llama/Llama-3.1-8B-Instruct",
    }

# ---------------------------------------------------------
# Ask Endpoint (RAG)
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User question to search and answer"),
    top_k: int = Query(3, ge=1, le=10, description="Number of top results"),
):
    """
    Step 1: Retrieve top-k FAQs  
    Step 2: Generate grounded answer using LLaMA-3.1-8B  
    Step 3: Return both retrieval context + generated answer
    """
    try:
        # 🔍 Step 1: Retrieve
        results_df = retriever.search(query, top_k=top_k)
        retrieved_docs = results_df["answer"].tolist()

        # 🧠 Step 2: Generate
        answer = generate_answer(query, retrieved_docs)

        # 📦 Step 3: Combine response
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
