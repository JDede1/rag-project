"""
search_engine.py
-------------------------------------------------------
FAISS-based semantic retriever for chunked RBC FAQ data.

Phase 4 Requirements:
    • Use upgraded all-mpnet-base-v2 embeddings
    • Retrieve chunk-level results (not whole answers)
    • Return structured, JSON-friendly dictionaries
    • Include provenance metadata (url, source, retrieved_at)
    • Normalize query embeddings for cosine similarity
    • Serve as backend utility for FastAPI

Usage:
    from retrieval.search_engine import RbcRetriever
    retriever = RbcRetriever()
    results = retriever.search("How do I report a lost credit card?", top_k=5)
"""

import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


class RbcRetriever:
    def __init__(self):
        """Load FAISS index, metadata, and embedding model once on startup."""
        base_dir = Path(__file__).resolve().parents[2]
        index_dir = base_dir / "data" / "index"

        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

        # Load index + metadata
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        # Load transformer model
        self.model = SentenceTransformer(self.model_name)

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"Loaded metadata rows: {len(self.metadata)}")

    def search(self, query: str, top_k: int = 5):
        """
        Perform semantic retrieval.

        Returns:
            List[dict] where each element contains:
                - question
                - chunk
                - score
                - url (if present)
                - source
                - retrieved_at
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query text cannot be empty.")

        # Encode search query
        query_emb = self.model.encode([query], convert_to_numpy=True)
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
                "source_faq_index": int(row.get("source_faq_index", -1))
            }

            results.append(entry)

        # Sort by score descending
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results

    def pretty_print(self, query: str, top_k: int = 5):
        """Developer utility for console testing."""
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
