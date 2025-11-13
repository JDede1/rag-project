"""
build_faiss_index.py
-------------------------------------------------------
Build a FAISS index for RBC FAQ embeddings generated using
the 'all-mpnet-base-v2' model. This index supports:

    • High-accuracy cosine similarity retrieval
    • Chunk-level metadata (question + chunk)
    • Full provenance:
        - source_faq_index
        - url
        - source
        - retrieved_at

Outputs:
    • rbc_faiss.index
    • Console test retrieval with explanations
"""

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_INDEX = BASE_DIR / "data" / "index"

EMBEDDINGS_PATH = DATA_INDEX / "rbc_embeddings.npy"
METADATA_PATH = DATA_INDEX / "rbc_metadata.parquet"
INDEX_PATH = DATA_INDEX / "rbc_faiss.index"

# Upgraded embedding model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
def build_faiss_index():
    print("Loading embeddings and metadata...")
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_parquet(METADATA_PATH)

    print(f"Embeddings shape: {embeddings.shape}")
    dim = embeddings.shape[1]  # Should be 768 for mpnet

    # ---------------------------------------------------
    # Initialize FAISS index (cosine similarity)
    # ---------------------------------------------------
    print("Creating FAISS index (cosine similarity)...")
    index = faiss.IndexFlatIP(dim)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Add vectors to the index
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to FAISS index")

    # Save the index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS index → {INDEX_PATH}")

    # ---------------------------------------------------
    # Test retrieval using mpnet
    # ---------------------------------------------------
    print("\nRunning test retrieval...")
    model = SentenceTransformer(MODEL_NAME)

    query = "How do I report a lost credit card?"
    print(f"Query: {query}")

    # GPU-aware encoding
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        device="cuda" if model.device is not None else None
    )

    faiss.normalize_L2(query_emb)

    # Retrieve top results
    k = 3
    D, I = index.search(query_emb, k=k)

    print(f"\nTop {k} most similar chunks:")
    for rank, idx in enumerate(I[0]):
        row = metadata.iloc[idx]
        score = float(D[0][rank])

        print(f"{rank + 1}. Score: {score:.4f}")
        print(f"Question: {row['question']}")
        print(f"Chunk: {row['chunk'][:160]}...")

        if "url" in row:
            print(f"URL: {row['url']}")
        if "source" in row:
            print(f"Source: {row['source']}")
        if "retrieved_at" in row:
            print(f"Retrieved at: {row['retrieved_at']}")

        print("")


if __name__ == "__main__":
    build_faiss_index()
