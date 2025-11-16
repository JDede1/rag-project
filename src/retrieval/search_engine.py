"""
search_engine.py
-------------------------------------------------------
FAISS-based semantic retriever for chunked RBC FAQ data.

Corrected Version (Stable):
    • Uses SAME embedding model as Phase 3 → MPNet (768-D)
    • Loads FAISS + metadata from /data/index/
    • Computes MPNet embeddings for queries
    • Normalizes vectors for cosine similarity
    • Returns JSON-safe structured dicts with provenance
"""

import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


class RbcRetriever:
    def __init__(self):
        """Load FAISS index, metadata, and MPNet embedding model."""

        # Phase 3 directory structure
        base_dir = Path(__file__).resolve().parents[2]
        index_dir = base_dir / "data" / "index"

        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        # IMPORTANT: Must match Phase 3 embeddings EXACTLY
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

        # Load FAISS + metadata
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        # Load MPNet (Phase 3 embedding model)
        self.model = SentenceTransformer(self.model_name)

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"Loaded metadata rows: {len(self.metadata)}")
        print(f"Retriever model: {self.model_name}")

    # ---------------------------------------------------------
    # Encode query using MPNet
    # ---------------------------------------------------------
    def embed_query(self, text: str):
        """Generate MPNet embedding for a single query string."""
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False  # We'll normalize manually
        )

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5):
        """
        Perform semantic retrieval.

        Returns list of dicts:
            - question
            - chunk
            - score
            - url
            - source
            - retrieved_at
            - source_faq_index
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query cannot be empty.")

        # Encode query
        query_emb = self.embed_query(query)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_emb)

        # Retrieve nearest neighbors
        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            row = self.metadata.iloc[idx]

            entry = {
                "question": row.get("question", ""),
                "chunk": row.get("chunk", ""),
                "score": float(score),
                "url": row.get("url", None),
                "source": row.get("source", None),
                "retrieved_at": row.get("retrieved_at", None),
                "source_faq_index": int(row.get("source_faq_index", -1)),
            }

            results.append(entry)

        # Sort results by descending similarity score
        return sorted(results, key=lambda r: r["score"], reverse=True)

    # ---------------------------------------------------------
    # Developer pretty-print helper
    # ---------------------------------------------------------
    def pretty_print(self, query: str, top_k: int = 5):
        results = self.search(query, top_k)
        print(f"\nQuery: {query}\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. ({r['score']:.4f}) {r['question']}")
            print(f"   Chunk: {r['chunk'][:160]}...")
            if r["url"]:
                print(f"   URL: {r['url']}")
            print("")


if __name__ == "__main__":
    retriever = RbcRetriever()
    retriever.pretty_print("How do I report a lost credit card?", top_k=5)
