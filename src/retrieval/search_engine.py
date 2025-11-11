"""
search_engine.py
-------------------------------------
Reusable FAISS-based semantic retriever for RBC FAQs.

Purpose:
    • Load FAISS index, metadata, and model
    • Embed user queries
    • Retrieve top-k semantically similar FAQs
    • Return structured results (question, answer, score, url)

Usage Example:
    from retrieval.search_engine import RbcRetriever

    retriever = RbcRetriever()
    results = retriever.search("How do I report a lost credit card?", top_k=3)
    print(results)
"""

import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer


class RbcRetriever:
    def __init__(self):
        """Initialize retriever — load FAISS index, metadata, and model."""
        base_dir = Path(__file__).resolve().parents[2]
        self.data_dir = base_dir / "data" / "index"

        self.index_path = self.data_dir / "rbc_faiss.index"
        self.meta_path = self.data_dir / "rbc_metadata.parquet"
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        print("🔹 Loading FAISS index and metadata...")
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)
        self.model = SentenceTransformer(self.model_name)

        print(f"✅ Loaded index with {self.index.ntotal} vectors.")
        print(f"✅ Loaded metadata with {len(self.metadata)} records.\n")

    def search(self, query: str, top_k: int = 3) -> pd.DataFrame:
        """Search FAISS index for top-k semantically similar entries."""
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty.")

        # Encode and normalize the query embedding
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)

        # Search top-k results
        D, I = self.index.search(query_emb, top_k)

        # Build results DataFrame
        results = self.metadata.iloc[I[0]].copy()
        results["score"] = D[0]
        results = results[["question", "answer", "url", "score"]]
        return results.sort_values(by="score", ascending=False).reset_index(drop=True)

    def pretty_print(self, query: str, top_k: int = 3):
        """Convenience function to display top-k results nicely."""
        print(f"\n🔍 Query: {query}\n")
        results = self.search(query, top_k)
        for i, row in results.iterrows():
            print(f"{i+1}. ({row['score']:.3f}) {row['question']}")
            print(f"   → {row['answer'][:200]}...")
            print(f"   📎 {row['url']}\n")
        print("—" * 70)


if __name__ == "__main__":
    retriever = RbcRetriever()
    retriever.pretty_print("How do I report a lost credit card?", top_k=3)
