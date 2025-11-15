"""
search_engine.py
-------------------------------------------------------
FAISS-based semantic retriever for chunked RBC FAQ data.

Updated:
    • Uses SAME embedding model as Phase 3 (Phi-3.5-Mini-Instruct)
    • Loads FAISS + metadata directly from /data/index/
    • Computes Phi embeddings for queries
    • Normalizes vectors for cosine similarity
    • Returns JSON-safe dicts with provenance
"""

import faiss
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel


class RbcRetriever:
    def __init__(self):
        """Load FAISS index, metadata, and Phi embedding model once at backend startup."""

        # Phase 3 directory structure
        base_dir = Path(__file__).resolve().parents[2]
        index_dir = base_dir / "data" / "index"

        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        # Must match Phase 3
        self.model_name = "microsoft/Phi-3.5-mini-instruct"

        # Load FAISS + metadata
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        # Load Phi model (same one used for Phase 3 embeddings)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"Loaded metadata rows: {len(self.metadata)}")
        print(f"Retriever model: {self.model_name}")

    # ---------------------------------------------------------
    # Encode query using Phi-3.5 (same as Phase 3 embeddings)
    # ---------------------------------------------------------
    def embed_query(self, text: str):
        """Generate Phi embedding for a single query string."""

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            model_out = self.model(**encoded)

        # Mean pooling
        last_hidden = model_out.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)

        masked = last_hidden * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pool = summed / counts

        # Convert to NumPy
        emb = mean_pool.cpu().numpy()

        return emb

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

        # Sort and return
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
