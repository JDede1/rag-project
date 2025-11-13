"""
build_faiss_index.py
-------------------------------------------------------
Build a FAISS index for RBC FAQ embeddings with provenance support.

This version matches the upgraded preprocessing + embedding pipeline:
    • Uses chunk-level metadata
    • Supports traceability via source_faq_index, url, source, retrieved_at
    • Performs cosine similarity search (L2-normalized vectors)

Outputs:
    • rbc_faiss.index       (FAISS vector index)
    • Console test retrieval for verification
"""

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_INDEX = BASE_DIR / "data" / "index"

EMBEDDINGS_PATH = DATA_INDEX / "rbc_embeddings.npy"
METADATA_PATH = DATA_INDEX / "rbc_metadata.parquet"
INDEX_PATH = DATA_INDEX / "rbc_faiss.index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
def build_faiss_index():
    print("Loading embeddings and metadata...")
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_parquet(METADATA_PATH)

    print(f"Embeddings shape: {embeddings.shape}")
    dim = embeddings.shape[1]

    # ---------------------------------------------------
    # Initialize FAISS index (cosine similarity)
    # ---------------------------------------------------
    print("Creating FAISS index (cosine similarity)...")
    index = faiss.IndexFlatIP(dim)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add vectors
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to FAISS index")

    # Save index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS index → {INDEX_PATH}")

    # ---------------------------------------------------
    # Test search
    # ---------------------------------------------------
    print("\nRunning test retrieval...")

    model = SentenceTransformer(MODEL_NAME)

    query = "How do I report a lost credit card?"
    print(f"Query: {query}")

    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    # Retrieve top 3 matches
    D, I = index.search(query_emb, k=3)

    print("\nTop 3 most similar chunks:")
    for rank, idx in enumerate(I[0]):
        row = metadata.iloc[idx]
        score = float(D[0][rank])

        print(f"{rank + 1}. Score: {score:.4f}")
        print(f"Question: {row['question']}")
        print(f"Chunk: {row['chunk'][:160]}...")
        
        # Optional provenance
        if "url" in row:
            print(f"URL: {row['url']}")
        if "source" in row:
            print(f"Source: {row['source']}")
        if "retrieved_at" in row:
            print(f"Retrieved at: {row['retrieved_at']}")

        print("")


if __name__ == "__main__":
    build_faiss_index()
