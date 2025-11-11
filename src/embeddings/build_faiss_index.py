"""
build_faiss_index.py
-------------------------------------
Build and test a FAISS index for RBC FAQ embeddings.

Purpose:
    • Load precomputed embeddings and metadata
    • Build FAISS index for fast similarity search
    • Save the index to disk for later API use
    • Run a sample semantic query test

Usage:
    python src/embeddings/build_faiss_index.py
"""

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_INDEX = BASE_DIR / "data" / "index"

EMBEDDINGS_PATH = DATA_INDEX / "rbc_embeddings.npy"
METADATA_PATH = DATA_INDEX / "rbc_metadata.parquet"
INDEX_PATH = DATA_INDEX / "rbc_faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def build_faiss_index():
    print("🔹 Loading embeddings and metadata...")
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_parquet(METADATA_PATH)

    print(f"✅ Embeddings shape: {embeddings.shape}")
    dim = embeddings.shape[1]

    # Initialize FAISS index (using cosine similarity)
    print("⚙️ Creating FAISS index...")
    index = faiss.IndexFlatIP(dim)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add embeddings to index
    index.add(embeddings)
    print(f"✅ Added {index.ntotal} vectors to FAISS index")

    # Save index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"💾 Saved FAISS index → {INDEX_PATH}")

    # -----------------------------
    # Test: Run a sample query
    # -----------------------------
    model = SentenceTransformer(MODEL_NAME)

    query = "How do I report a lost credit card?"
    print(f"\n🔍 Testing query: '{query}'")

    query_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    # Retrieve top 3 results
    D, I = index.search(query_emb, k=3)

    print("\n📊 Top 3 most similar FAQs:")
    for rank, idx in enumerate(I[0]):
        q = metadata.iloc[idx]["question"]
        a = metadata.iloc[idx]["answer"][:150] + "..."
        score = float(D[0][rank])
        print(f"{rank+1}. ({score:.3f}) {q}\n   → {a}\n")


if __name__ == "__main__":
    build_faiss_index()
