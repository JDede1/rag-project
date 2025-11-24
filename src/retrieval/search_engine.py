"""
search_engine.py — Phase 6 Enhanced Retriever
-------------------------------------------------------
Adds:
    • Stage-2 heuristic re-ranking
    • Query–chunk keyword scoring
    • Citation IDs for generator.py
    • Confidence scoring for main.py fallback

Phase 7.2 Ready:
    • Retrieval metadata fields remain unchanged
    • Safe to use for monitoring (grounding_score, context_overlap added later in main.py)
"""

import faiss
import numpy as np
import pandas as pd
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Utility: Simple keyword tokenizer
# ---------------------------------------------------------
def _tokenize(text: str):
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


QUESTION_WORDS = {"how", "what", "when", "where", "why", "who", "does", "do", "can"}


class RbcRetriever:
    def __init__(self):
        """Load FAISS index, metadata, and MPNet embedding model."""

        # Phase 3 folder layout
        base_dir = Path(__file__).resolve().parents[2]
        index_dir = base_dir / "data" / "index"

        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        # Must match Phase 3 exactly
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

        # Load FAISS + metadata
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        # Load MPNet encoder
        self.model = SentenceTransformer(self.model_name)

        print(f"[Retriever] Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"[Retriever] Loaded metadata rows: {len(self.metadata)}")
        print(f"[Retriever] Embedding model: {self.model_name}")

    # ---------------------------------------------------------
    # Encode query
    # ---------------------------------------------------------
    def embed_query(self, text: str):
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False
        )

    # ---------------------------------------------------------
    # Stage-2 Reranking (Heuristic)
    # ---------------------------------------------------------
    def _rerank(self, query: str, results: list):
        """
        Lightweight reranking:
            • FAISS cosine score (base)
            • Keyword overlap
            • Question-word presence
        """

        q_tokens = set(_tokenize(query))
        question_overlap = q_tokens & QUESTION_WORDS

        reranked = []
        for i, r in enumerate(results):
            chunk_tokens = set(_tokenize(r["chunk"]))

            # Score components
            overlap = len(q_tokens & chunk_tokens)
            base = r["score"]
            qword_boost = 2 if question_overlap else 1

            final_score = base + 0.02 * overlap * qword_boost

            r["final_score"] = float(final_score)
            r["citation_id"] = i + 1  # stable sequential IDs for generator

            reranked.append(r)

        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)

    # ---------------------------------------------------------
    # Main Search
    # ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5):
        if not query or not isinstance(query, str):
            raise ValueError("Query cannot be empty.")

        # Step 1 — encode
        query_emb = self.embed_query(query)
        faiss.normalize_L2(query_emb)

        # Step 2 — FAISS search
        distances, indices = self.index.search(query_emb, top_k)

        raw_results = []
        for score, idx in zip(distances[0], indices[0]):
            row = self.metadata.iloc[idx]

            raw_results.append({
                "question": row.get("question", ""),
                "chunk": row.get("chunk", ""),
                "score": float(score),
                "url": row.get("url", None),
                "source": row.get("source", None),
                "retrieved_at": row.get("retrieved_at", None),
                "source_faq_index": int(row.get("source_faq_index", -1)),
            })

        # Step 3 — Stage-2 reranking
        reranked = self._rerank(query, raw_results)

        # Step 4 — Confidence scoring (shared for entire retrieved set)
        if len(reranked) >= 2:
            top_two = reranked[:2]
            avg = (top_two[0]["final_score"] + top_two[1]["final_score"]) / 2
        else:
            avg = reranked[0]["final_score"] if reranked else 0.0

        avg = float(avg)

        for r in reranked:
            r["confidence"] = avg  # stable confidence for the group

        return reranked

    # ---------------------------------------------------------
    # Pretty Printer (unchanged)
    # ---------------------------------------------------------
    def pretty_print(self, query: str, top_k: int = 5):
        results = self.search(query, top_k)
        print(f"\nQuery: {query}\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. ({r['final_score']:.4f}) {r['question']}")
            print(f"   Chunk: {r['chunk'][:160]}...")
            if r["url"]:
                print(f"   URL: {r['url']}")
            print(f"   CIT: {r['citation_id']}")
            print("")


if __name__ == "__main__":
    retriever = RbcRetriever()
    retriever.pretty_print("How do I report a lost credit card?", top_k=5)
