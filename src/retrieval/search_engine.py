"""
Hybrid Search Engine — MPNet (Local) and ONNX (Cloud)
---------------------------------------------------------------------------
Modes:
    • Local (default) — SentenceTransformer MPNet (full accuracy)
    • Cloud Run — ONNX Runtime MPNet encoder (fast, lightweight, no HF downloads)

Selection:
    Set DEPLOY_ENV=cloud in Cloud Run
    Otherwise uses local MPNet.

This file preserves:
    • FAISS high-recall search (30 candidates)
    • Topic-aware reranking
    • Lexical overlap + question word boosts
    • Confidence scoring
    • Identical output structure (no breaking changes)
"""

import os
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

# Local MPNet (optional on cloud)
from sentence_transformers import SentenceTransformer

# ONNX encoder (for Cloud Run)
import onnxruntime as ort


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
# Hybrid Retriever (Local MPNet + Cloud ONNX)
# =========================================================
class RbcRetriever:
    def __init__(self):
        # -----------------------------------------------------
        # Load FAISS + metadata
        # -----------------------------------------------------
        base_dir = Path(__file__).resolve().parents[2]
        index_dir = base_dir / "data" / "index"

        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata parquet not found: {self.meta_path}")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        self.faiss_dim = self.index.d
        print(f"[Retriever] FAISS index loaded ({self.index.ntotal} vectors)")

        # -----------------------------------------------------
        # Determine deployment mode
        # -----------------------------------------------------
        self.deploy_env = os.getenv("DEPLOY_ENV", "local").lower()

        if self.deploy_env == "cloud":
            print("[Retriever] DEPLOY_ENV=cloud → Using ONNX Runtime encoder")
            self._load_onnx_encoder()
        else:
            print("[Retriever] DEPLOY_ENV=local → Using SentenceTransformer MPNet")
            self._load_local_encoder()

    # =========================================================
    # LOCAL MODE — SentenceTransformer MPNet
    # =========================================================
    def _load_local_encoder(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)

        # Validate embedding dimension
        test_emb = self.model.encode(["test"], convert_to_numpy=True)
        if test_emb.shape[1] != self.faiss_dim:
            raise ValueError(
                f"Local MPNet dim {test_emb.shape[1]} does not match FAISS dim {self.faiss_dim}"
            )

        self.encoder_type = "mpnet_local"
        print(f"[Retriever] Local MPNet loaded: {model_name}")

    # =========================================================
    # CLOUD MODE — ONNX Runtime (MPNet)
    # =========================================================
    def _load_onnx_encoder(self):
        """
        Expects:
            data/index/mpnet.onnx
            data/index/tokenizer.json
        """
        base_dir = Path(__file__).resolve().parents[2]
        onnx_dir = base_dir / "data" / "index"

        self.onnx_path = onnx_dir / "mpnet.onnx"
        self.tokenizer_path = onnx_dir / "tokenizer.json"

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model missing: {self.onnx_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer missing: {self.tokenizer_path}")

        self.ort_session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"]
        )

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))

        # Validate output dimension
        dummy = self._encode_onnx_embeddings("test")
        if dummy.shape[1] != self.faiss_dim:
            raise ValueError(
                f"ONNX embedding dim {dummy.shape[1]} does not match FAISS dim {self.faiss_dim}"
            )

        self.encoder_type = "mpnet_onnx"
        print(f"[Retriever] ONNX encoder loaded: {self.onnx_path}")

    # =========================================================
    # ONNX embedding
    # =========================================================
    def _encode_onnx_embeddings(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True
        )

        ort_inputs = {k: v for k, v in tokens.items()}
        ort_out = self.ort_session.run(None, ort_inputs)[0]

        emb = ort_out.astype(np.float32)
        faiss.normalize_L2(emb)
        return emb

    # =========================================================
    # Hybrid interface — identical return output
    # =========================================================
    def embed_query(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            raise ValueError("Query must be a non-empty string.")

        if self.encoder_type == "mpnet_onnx":
            return self._encode_onnx_embeddings(text)
        else:
            emb = self.model.encode(
                [text], convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float32)
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

            lexical_boost = 0.12 * lexical_overlap
            qword_boost = 0.05 * (1 if question_overlap else 0)

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

    # =========================================================
    # MAIN SEARCH — High Recall (FAISS)
    # =========================================================
    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        RECALL_K = 30

        query_emb = self.embed_query(query)
        distances, indices = self.index.search(query_emb, RECALL_K)

        raw = []
        for score, idx in zip(distances[0], indices[0]):
            row = self.metadata.iloc[idx]

            raw.append(
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

        reranked = self._rerank(query, raw)
        clipped = reranked[:top_k]

        # Confidence = avg of top2 final scores
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
