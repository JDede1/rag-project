"""
search_engine.py — MPNet Retriever (SentenceTransformers)
---------------------------------------------------------
Production Retriever:
    • Uses sentence-transformers/all-mpnet-base-v2 for query embeddings
    • Loads FAISS index + metadata built in Phase 3
    • Performs vector search + lightweight reranking
    • Provides citation IDs + confidence scores
"""

import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
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
        """
        Load FAISS index, metadata, and MPNet encoder.

        Assumes Phase 3 has already created:
            data/index/rbc_faiss.index
            data/index/rbc_metadata.parquet

        FAISS index was built from all-mpnet-base-v2 embeddings
        (normalized, IndexFlatIP).
        """

        base_dir = Path(__file__).resolve().parents[2]

        # -----------------------------
        # Load FAISS + metadata
        # -----------------------------
        index_dir = base_dir / "data" / "index"
        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata parquet not found: {self.meta_path}")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        # -----------------------------
        # Load MPNet encoder
        # -----------------------------
        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)

        # Embedding dimension sanity check
        dim = self.index.d
        test_emb = self.model.encode(["test"], convert_to_numpy=True)
        if test_emb.shape[1] != dim:
            raise ValueError(
                f"Embedding dimension mismatch: FAISS index dim={dim}, "
                f"MPNet output dim={test_emb.shape[1]}"
            )

        print(f"[Retriever] Loaded FAISS index: {self.index.ntotal} vectors")
        print(f"[Retriever] Loaded metadata rows: {len(self.metadata)}")
        print(f"[Retriever] MPNet model loaded: {model_name}")

    # ---------------------------------------------------------
    # MPNet Encoder
    # ---------------------------------------------------------
    def embed_query(self, text: str) -> np.ndarray:
        """
        Encode text → 768-d MPNet embedding.

        We normalize embeddings to match Phase 3, where:
            - embeddings were normalized with faiss.normalize_L2
            - IndexFlatIP is used (cosine similarity).
        """

        if not text or not isinstance(text, str):
            raise ValueError("Query text must be a non-empty string.")

        emb = self.model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)

        # Normalize for cosine similarity with IndexFlatIP
        faiss.normalize_L2(emb)

        return emb

    # ---------------------------------------------------------
    # Stage-2 Reranking
    # ---------------------------------------------------------
    def _rerank(self, query: str, results: list):
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

        # Encode → search
        query_emb = self.embed_query(query)  # already normalized
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

        reranked = self._rerank(query, raw_results)

        if len(reranked) >= 2:
            avg = (reranked[0]["final_score"] + reranked[1]["final_score"]) / 2
        else:
            avg = reranked[0]["final_score"] if reranked else 0.0

        avg = float(avg)

        for r in reranked:
            r["confidence"] = avg

        return reranked

    # ---------------------------------------------------------
    # Pretty Print
    # ---------------------------------------------------------
    def pretty_print(self, query: str, top_k: int = 5):
        results = self.search(query, top_k)
        print(f"\nQuery: {query}\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. ({r['final_score']:.4f}) {r['question']}")
            print(f"   Chunk: {r['chunk'][:140]}...")
            if r.get("url"):
                print(f"   URL: {r['url']}")
            print(f"   CIT: {r['citation_id']}")
            print("")


if __name__ == "__main__":
    retriever = RbcRetriever()
    retriever.pretty_print("How do I report a lost credit card?", top_k=5)
