"""
search_engine.py — MPNet Retriever (High-Recall + Strong Reranking)
---------------------------------------------------------------------------
Upgrades:
    • FAISS search depth expanded to 30 for high recall
    • Strong topic-aware reranking (lostcard, fraud, login, etransfer)
    • Lexical overlap + question-word boosting
    • Deterministic and fully compatible with generator + evaluation

Produces:
    [
        {
            "question": ...,
            "chunk": ...,
            "score": <faiss score>,
            "final_score": <reranked score>,
            "citation_id": <int>,
            "topic": <topic label>,
            "confidence": <float>,
            ...
        }
    ]
"""

import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Utility Tokenizer
# ---------------------------------------------------------
def _tokenize(text: str):
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------
# Topic classifiers
# ---------------------------------------------------------
def classify_chunk_topic(text: str) -> str:
    t = text.lower()

    if any(k in t for k in ["lost", "stolen", "misplaced", "block your card"]):
        return "lostcard"

    if any(k in t for k in ["fraud", "unauthorized", "dispute"]):
        return "fraud"

    if any(k in t for k in ["password", "login", "reset", "passcode"]):
        return "login"

    if any(k in t for k in ["interac", "e-transfer", "etransfer", "transfer"]):
        return "etransfer"

    return "general"


def classify_query_topic(q: str) -> str:
    q = q.lower()

    if any(k in q for k in ["lost", "stolen", "misplaced"]):
        return "lostcard"

    if any(k in q for k in ["fraud", "unauthorized", "dispute"]):
        return "fraud"

    if any(k in q for k in ["password", "login", "reset"]):
        return "login"

    if any(k in q for k in ["interac", "e-transfer", "etransfer", "transfer"]):
        return "etransfer"

    return "general"


QUESTION_WORDS = {"how", "what", "when", "where", "why", "who", "does", "do", "can"}


# =========================================================
# RbcRetriever (FAISS + MPNet Encoder)
# =========================================================
class RbcRetriever:
    def __init__(self):
        base_dir = Path(__file__).resolve().parents[2]
        index_dir = base_dir / "data" / "index"

        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata parquet not found: {self.meta_path}")

        # Load index and metadata
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        # Load PyTorch MPNet encoder
        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)

        # Embedding sanity check
        dim = self.index.d
        test_emb = self.model.encode(["test"], convert_to_numpy=True)
        if test_emb.shape[1] != dim:
            raise ValueError(
                f"Embedding dimension mismatch: FAISS dim={dim}, MPNet dim={test_emb.shape[1]}"
            )

        print(f"[Retriever] Loaded FAISS index: {self.index.ntotal} vectors")
        print(f"[Retriever] Loaded metadata:    {len(self.metadata)} rows")
        print(f"[Retriever] MPNet model loaded: {model_name}")


    # ---------------------------------------------------------
    # Encode query with MPNet
    # ---------------------------------------------------------
    def embed_query(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            raise ValueError("Query text must be a non-empty string.")

        emb = (
            self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            .astype(np.float32)
        )
        faiss.normalize_L2(emb)
        return emb


    # ---------------------------------------------------------
    # Reranking
    # ---------------------------------------------------------
    def _rerank(self, query: str, results: list):
        q_tokens = set(_tokenize(query))
        q_topic = classify_query_topic(query)
        question_overlap = q_tokens & QUESTION_WORDS

        reranked = []

        for i, r in enumerate(results):
            chunk = r["chunk"]
            chunk_topic = classify_chunk_topic(chunk)

            chunk_tokens = set(_tokenize(chunk))
            lexical_overlap = len(q_tokens & chunk_tokens)

            base_score = r["score"]

            # 1. Strong lexical boost
            lexical_boost = 0.12 * lexical_overlap

            # 2. Mild question-word boost
            qword_boost = 0.05 * (1 if question_overlap else 0)

            # 3. Topic alignment boost (critical fix)
            topic_boost = 0.0
            if q_topic == chunk_topic:
                topic_boost = 0.30
            elif q_topic != "general" and chunk_topic != "general":
                topic_boost = -0.10

            final_score = float(base_score + lexical_boost + qword_boost + topic_boost)

            r["final_score"] = final_score
            r["citation_id"] = i + 1
            r["topic"] = chunk_topic

            reranked.append(r)

        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)


    # ---------------------------------------------------------
    # MAIN SEARCH — High-Recall Version
    # ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5):
        if not query or not isinstance(query, str):
            raise ValueError("Query cannot be empty.")

        # Expand FAISS recall (Option A)
        RECALL_K = 30

        query_emb = self.embed_query(query)
        distances, indices = self.index.search(query_emb, RECALL_K)

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

        # Rerank expanded results
        reranked = self._rerank(query, raw_results)

        # Clip final results
        clipped = reranked[:top_k]

        # Assign confidence = mean of top-2 final scores
        if len(clipped) >= 2:
            avg = (clipped[0]["final_score"] + clipped[1]["final_score"]) / 2
        else:
            avg = clipped[0]["final_score"] if clipped else 0.0

        avg = float(avg)
        for r in clipped:
            r["confidence"] = avg

        return clipped


    # ---------------------------------------------------------
    # Pretty Print Debug Utility
    # ---------------------------------------------------------
    def pretty_print(self, query: str, top_k: int = 5):
        results = self.search(query, top_k)
        print(f"\nQuery: {query}\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. ({r['final_score']:.4f}) [topic={r['topic']}] {r['question']}")
            print(f"   Chunk: {r['chunk'][:160]}...")
            if r.get("url"):
                print(f"   URL: {r['url']}")
            print(f"   CIT: {r['citation_id']}\n")


# Standalone Test
if __name__ == "__main__":
    retriever = RbcRetriever()
    retriever.pretty_print("How do I report a lost credit card?", top_k=5)
