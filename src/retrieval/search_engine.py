"""
search_engine.py — Phase 7 Stable Retriever
-------------------------------------------------------
Phase 6 Provided:
    • Stage-2 heuristic reranking
    • Keyword overlap scoring
    • Citation IDs for generator.py
    • Confidence scoring for main.py fallback

Phase 7:
    • No structural changes required
    • Retrieval output remains fully compatible with:
        - Phase 7 logging (rag_logger.py)
        - Phase 7 generator (grounding_details)
        - Phase 7 main.py monitoring fields
    • This file is now the stable production version
"""

import faiss
import numpy as np
import pandas as pd
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Utility: Simple tokenizer for keyword overlap
# ---------------------------------------------------------
def _tokenize(text: str):
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


QUESTION_WORDS = {"how", "what", "when", "where", "why", "who", "does", "do", "can"}


class RbcRetriever:
    def __init__(self):
        """Load FAISS index, metadata, and MPNet embedding model."""

        base_dir = Path(__file__).resolve().parents[2]
        index_dir = base_dir / "data" / "index"

        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        # MUST match Phase 3 embeddings exactly
        self.model_name = "sentence-transformers/all-mpnet-base-v2"

        # Load FAISS and metadata
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
    # Stage-2 Reranking
    # ---------------------------------------------------------
    def _rerank(self, query: str, results: list):
        """
        Lightweight reranking:
            - base FAISS cosine score
            - keyword overlap
            - question-word boost
            - citation IDs added for generator
        """

        q_tokens = set(_tokenize(query))
        question_overlap = q_tokens & QUESTION_WORDS

        reranked = []
        for i, r in enumerate(results):
            chunk_tokens = set(_tokenize(r["chunk"]))
            overlap = len(q_tokens & chunk_tokens)

            base = r["score"]
            qword_boost = 2 if question_overlap else 1

            final_score = base + 0.02 * overlap * qword_boost

            r["final_score"] = float(final_score)
            r["citation_id"] = i + 1

            reranked.append(r)

        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)

    # ---------------------------------------------------------
    # Main Search
    # ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5):
        if not query or not isinstance(query, str):
            raise ValueError("Query cannot be empty.")

        # Encode query
        query_emb = self.embed_query(query)
        faiss.normalize_L2(query_emb)

        # FAISS search
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

        # Rerank
        reranked = self._rerank(query, raw_results)

        # Confidence = avg of top 2 final_scores
        if len(reranked) >= 2:
            top_two = reranked[:2]
            avg = (top_two[0]["final_score"] + top_two[1]["final_score"]) / 2
        else:
            avg = reranked[0]["final_score"] if reranked else 0.0

        avg = float(avg)

        for r in reranked:
            r["confidence"] = avg

        return reranked

    # ---------------------------------------------------------
    # Pretty Printer (Developer Debug Tool)
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
