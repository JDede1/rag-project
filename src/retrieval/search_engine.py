"""
search_engine.py — ONNX-based MPNet Retriever
-------------------------------------------------------
Production Retriever:
    • Loads ONNX MPNet encoder (fast, CPU optimized)
    • Loads FAISS index + metadata
    • Performs vector search with reranking
    • Provides citation IDs + confidence scores
"""

import faiss
import numpy as np
import pandas as pd
import re
from pathlib import Path
import onnxruntime as ort


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
        """Load FAISS index, metadata, and ONNX MPNet model."""

        base_dir = Path(__file__).resolve().parents[2]

        # -----------------------------
        # Load FAISS + metadata
        # -----------------------------
        index_dir = base_dir / "data" / "index"
        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        # -----------------------------
        # Load ONNX MPNet model + tokenizer files
        # -----------------------------
        model_dir = base_dir / "models" / "mpnet"
        self.onnx_path = model_dir / "model.onnx"
        self.model_dir = model_dir

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"]
        )

        # Tokenizer files loaded from disk
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        print(f"[Retriever] Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"[Retriever] Loaded metadata rows: {len(self.metadata)}")
        print(f"[Retriever] ONNX model: {self.onnx_path.name}")

    # ---------------------------------------------------------
    # ONNX Encoder
    # ---------------------------------------------------------
    def embed_query(self, text: str):
        """Encode query text using ONNX MPNet encoder."""
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        outputs = self.session.run(["last_hidden_state"], ort_inputs)
        # Mean pooling for SentenceTransformer-compatible embeddings
        token_embeddings = outputs[0]  # shape: (1, seq, 768)
        attention_mask = inputs["attention_mask"]

        mask = attention_mask[..., None]
        summed = (token_embeddings * mask).sum(axis=1)
        counts = mask.sum(axis=1)

        sentence_embedding = summed / np.clip(counts, a_min=1e-9, a_max=None)
        return sentence_embedding.astype(np.float32)

    # ---------------------------------------------------------
    # Stage-2 Reranking
    # ---------------------------------------------------------
    def _rerank(self, query: str, results: list):
        """
        Lightweight reranking:
            - FAISS cosine similarity score
            - keyword overlap
            - question-word boost
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

        # Encode query → normalize → search
        query_emb = self.embed_query(query)
        faiss.normalize_L2(query_emb)

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

        reranked = self._rerank(query, raw_results)

        # Confidence = average of top 2 scores
        if len(reranked) >= 2:
            avg = (reranked[0]["final_score"] + reranked[1]["final_score"]) / 2
        else:
            avg = reranked[0]["final_score"] if reranked else 0.0

        avg = float(avg)

        for r in reranked:
            r["confidence"] = avg

        return reranked

    # ---------------------------------------------------------
    # Developer Pretty Printer
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
