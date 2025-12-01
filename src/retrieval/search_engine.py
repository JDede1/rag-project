"""
search_engine.py — MPNet Retriever (PyTorch SentenceTransformers)
----------------------------------------------------------------
Production Retriever:
    • Uses sentence-transformers/all-mpnet-base-v2 for query embeddings
    • Loads FAISS index + metadata built in Phase 3
    • Performs vector search + stable reranking
    • Provides citation IDs + confidence scores
    • Fully consistent with generator.py (Option A) and main.py
"""

import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Utility Tokenizer (Light, Safe)
# ---------------------------------------------------------
def _tokenize(text: str):
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


# Question words boost reranking weight (mild heuristic)
QUESTION_WORDS = {"how", "what", "when", "where", "why", "who", "does", "do", "can"}


# =========================================================
# RETRIEVER CLASS (Pytorch MPNet + FAISS)
# =========================================================
class RbcRetriever:
    def __init__(self):
        """
        Loads:
            - FAISS index: data/index/rbc_faiss.index
            - Metadata:    data/index/rbc_metadata.parquet
            - PyTorch MPNet: all-mpnet-base-v2
        
        Notes:
            • Matches EXACT same embedding model used in Phase 3.
            • Embeddings are L2-normalized before FAISS search.
            • IndexFlatIP gives cosine similarity.
        """

        base_dir = Path(__file__).resolve().parents[2]

        index_dir = base_dir / "data" / "index"
        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata parquet not found: {self.meta_path}")

        # Load FAISS index + metadata
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        # Load PyTorch MPNet encoder
        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)

        # Sanity check embedding dimension
        dim = self.index.d
        test_emb = self.model.encode(["test"], convert_to_numpy=True)
        if test_emb.shape[1] != dim:
            raise ValueError(
                f"Embedding dimension mismatch: FAISS dim={dim}, "
                f"MPNet dim={test_emb.shape[1]}"
            )

        print(f"[Retriever] Loaded FAISS index: {self.index.ntotal} vectors")
        print(f"[Retriever] Loaded metadata:    {len(self.metadata)} rows")
        print(f"[Retriever] MPNet model loaded: {model_name}")


    # ---------------------------------------------------------
    # Embed Query (PyTorch MPNet)
    # ---------------------------------------------------------
    def embed_query(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            raise ValueError("Query text must be a non-empty string.")

        emb = self.model.encode(
            [text], convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(emb)

        return emb


    # ---------------------------------------------------------
    # Stage-2 Reranking (Stable, Predictable)
    # ---------------------------------------------------------
    def _rerank(self, query: str, results: list):
        """
        Lightweight reranking:
            • Preserves FAISS score dominance
            • Adds mild keyword overlap boost
            • Ensures lost/stolen chunks don't get overshadowed by fraud chunks
        """
        q_tokens = set(_tokenize(query))
        question_overlap = q_tokens & QUESTION_WORDS

        reranked = []
        for i, r in enumerate(results):
            chunk_tokens = set(_tokenize(r["chunk"]))
            overlap = len(q_tokens & chunk_tokens)

            base_score = r["score"]
            boost = 0.02 * overlap * (2 if question_overlap else 1)

            final_score = float(base_score + boost)

            r["final_score"] = final_score
            r["citation_id"] = i + 1  # sequentially assigned
            reranked.append(r)

        # Highest final_score first
        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)


    # ---------------------------------------------------------
    # MAIN SEARCH FUNCTION
    # ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5):
        if not query or not isinstance(query, str):
            raise ValueError("Query cannot be empty.")

        # Encode → search FAISS
        query_emb = self.embed_query(query)
        distances, indices = self.index.search(query_emb, top_k)

        raw_results = []
        for score, idx in zip(distances[0], indices[0]):
            row = self.metadata.iloc[idx]

            raw_results.append(
                {
                    "question": row.get("question", ""),
                    "chunk": row.get("chunk", ""),
                    "score": float(score),
                    "url": row.get("url"),
                    "source": row.get("source"),
                    "retrieved_at": row.get("retrieved_at"),
                    "source_faq_index": int(row.get("source_faq_index", -1)),
                }
            )

        # Stable rerank
        reranked = self._rerank(query, raw_results)

        # Confidence = mean of top-2 scores
        if len(reranked) >= 2:
            avg = (reranked[0]["final_score"] + reranked[1]["final_score"]) / 2
        else:
            avg = reranked[0]["final_score"] if reranked else 0.0

        avg = float(avg)

        for r in reranked:
            r["confidence"] = avg

        return reranked


    # ---------------------------------------------------------
    # Pretty Print Debug Utility
    # ---------------------------------------------------------
    def pretty_print(self, query: str, top_k: int = 5):
        results = self.search(query, top_k)
        print(f"\nQuery: {query}\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. ({r['final_score']:.4f}) {r['question']}")
            print(f"   Chunk: {r['chunk'][:140]}...")
            if r.get("url"):
                print(f"   URL: {r['url']}")
            print(f"   CIT: {r['citation_id']}\n")


# Standalone Test
if __name__ == "__main__":
    retriever = RbcRetriever()
    retriever.pretty_print("How do I report a lost credit card?", top_k=5)
