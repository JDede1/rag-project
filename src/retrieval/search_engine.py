"""
Hybrid Search Engine — Dual MiniLM ONNX Support
------------------------------------------------------

Modes:

    • Local (default)
        - SentenceTransformer MiniLM encoder
        - OR Local ONNX encoder (minilm_local.onnx, IR 10)

    • Cloud Run (production)
        - Cloud-safe ONNX encoder (minilm.onnx, IR ≤ 9)

Mode selection:

    DEPLOY_ENV=cloud → Cloud Run ONNX mode (minilm.onnx)
    Default          → Local ST MiniLM or local ONNX (minilm_local.onnx)
"""

import os
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# CLOUD / LOCAL detection
# ---------------------------------------------------------
IS_CLOUD = os.getenv("DEPLOY_ENV", "").lower() == "cloud"

if not IS_CLOUD:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None
else:
    SentenceTransformer = None


def _tokenize(text: str):
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


# ======================================================================
# TOPIC CLASSIFICATION
# ======================================================================
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


# ======================================================================
# HYBRID RETRIEVER
# ======================================================================
class RbcRetriever:
    def __init__(self):

        if IS_CLOUD:
            base_dir = Path("/app")
        else:
            base_dir = Path(__file__).resolve().parents[2]

        index_dir = base_dir / "data" / "index"

        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        # NEW: Dual ONNX paths
        self.onnx_path_cloud = index_dir / "minilm.onnx"           # Cloud-safe IR <= 9
        self.onnx_path_local = index_dir / "minilm_local.onnx"     # Local IR 10 ONNX

        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.meta_path}")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)
        self.faiss_dim = self.index.d

        print(f"[Retriever] Loaded FAISS index ({self.index.ntotal} vectors, dim={self.faiss_dim})")

        # MODE SELECTION
        if IS_CLOUD:
            print("[Retriever] Cloud mode → Cloud ONNX MiniLM (minilm.onnx)")
            self._load_cloud_onnx_encoder()

        else:
            # Local: prefer local ONNX if available
            if self.onnx_path_local.exists():
                print("[Retriever] Local mode → Local ONNX MiniLM (minilm_local.onnx)")
                self._load_local_onnx_encoder()
            else:
                print("[Retriever] Local mode → SentenceTransformer MiniLM")
                self._load_local_encoder()

    # ==================================================================
    # LOCAL ST MODE
    # ==================================================================
    def _load_local_encoder(self):

        if SentenceTransformer is None:
            raise RuntimeError("SentenceTransformer is not available.")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)

        test_emb = self.model.encode(["test"], convert_to_numpy=True)
        if test_emb.shape[1] != self.faiss_dim:
            raise ValueError(
                f"Local MiniLM dim {test_emb.shape[1]} != FAISS dim {self.faiss_dim}"
            )

        self.encoder_type = "minilm_local_st"
        print("[Retriever] SentenceTransformer MiniLM loaded.")

    # ==================================================================
    # CLOUD ONNX (IR <= 9)
    # ==================================================================
    def _load_cloud_onnx_encoder(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not self.onnx_path_cloud.exists():
            raise FileNotFoundError(f"Cloud ONNX missing: {self.onnx_path_cloud}")

        onnx_dir = self.onnx_path_cloud.parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir), local_files_only=True)
        self.ort_session = ort.InferenceSession(str(self.onnx_path_cloud))

        emb = self._encode_onnx_embeddings("test")
        if emb.shape[1] != self.faiss_dim:
            raise ValueError("Cloud ONNX embedding dimension mismatch")

        self.encoder_type = "minilm_cloud_onnx"
        print("[Retriever] Cloud ONNX MiniLM loaded.")

    # ==================================================================
    # LOCAL ONNX (IR 10)
    # ==================================================================
    def _load_local_onnx_encoder(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not self.onnx_path_local.exists():
            raise FileNotFoundError(f"Local ONNX missing: {self.onnx_path_local}")

        onnx_dir = self.onnx_path_local.parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir), local_files_only=True)
        self.ort_session = ort.InferenceSession(str(self.onnx_path_local))

        emb = self._encode_onnx_embeddings("test")
        if emb.shape[1] != self.faiss_dim:
            raise ValueError("Local ONNX embedding dimension mismatch")

        self.encoder_type = "minilm_local_onnx"
        print("[Retriever] Local ONNX MiniLM loaded.")

    # ==================================================================
    # ONNX encoding
    # ==================================================================
    def _encode_onnx_embeddings(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=256,
        )

        ort_out = self.ort_session.run(None, tokens)[0]

        if ort_out.ndim == 3:
            emb = ort_out.mean(axis=1)
        else:
            emb = ort_out

        emb = emb.astype(np.float32)
        if emb.ndim == 1:
            emb = np.expand_dims(emb, 0)

        faiss.normalize_L2(emb)
        return emb

    # ==================================================================
    # EMBEDDING ROUTER
    # ==================================================================
    def embed_query(self, text: str):
        if "onnx" in self.encoder_type:
            return self._encode_onnx_embeddings(text)

        emb = self.model.encode([text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(emb)
        return emb

    # ==================================================================
    # RERANK
    # ==================================================================
    def _rerank(self, query, results):

        q_tokens = set(_tokenize(query))
        q_topic = classify_query_topic(query)
        has_q_word = bool(q_tokens & QUESTION_WORDS)

        reranked = []

        for i, r in enumerate(results):

            chunk = r["chunk"]
            chunk_topic = classify_chunk_topic(chunk)

            chunk_tokens = set(_tokenize(chunk))
            lexical_overlap = len(q_tokens & chunk_tokens)

            base = r["score"]
            lexical_boost = 0.12 * lexical_overlap
            question_boost = 0.05 if has_q_word else 0

            if q_topic == chunk_topic:
                topic_boost = 0.30
            elif q_topic != "general" and chunk_topic != "general":
                topic_boost = -0.10
            else:
                topic_boost = 0

            final = float(base + lexical_boost + question_boost + topic_boost)

            r["final_score"] = final
            r["citation_id"] = i + 1
            r["topic"] = chunk_topic

            reranked.append(r)

        return sorted(reranked, key=lambda x: x["final_score"], reverse=True)

    # ==================================================================
    # MAIN SEARCH
    # ==================================================================
    def search(self, query: str, top_k: int = 5):
        if not query.strip():
            raise ValueError("Empty query")

        RECALL_K = 30

        q_emb = self.embed_query(query)
        distances, indices = self.index.search(q_emb, RECALL_K)

        raw = []
        for score, idx in zip(distances[0], indices[0]):
            row = self.metadata.iloc[idx]
            raw.append({
                "question": row.get("question", ""),
                "chunk": row.get("chunk", ""),
                "score": float(score),
                "url": row.get("url"),
                "source": row.get("source"),
                "retrieved_at": row.get("retrieved_at"),
                "source_faq_index": int(row.get("source_faq_index", -1)),
            })

        reranked = self._rerank(query, raw)
        clipped = reranked[:top_k]

        if len(clipped) >= 2:
            avg = (clipped[0]["final_score"] + clipped[1]["final_score"]) / 2
        else:
            avg = clipped[0]["final_score"]

        for r in clipped:
            r["confidence"] = float(avg)

        return clipped

    # ==================================================================
    # PRINT
    # ==================================================================
    def pretty_print(self, query, top_k=5):
        results = self.search(query, top_k)
        print(f"\nQuery: {query}\n")
        for r in results:
            print(f"{r['citation_id']}. [{r['topic']}] score={r['final_score']:.4f}")
            print("   Q:", r["question"])
            print("   Chunk:", r["chunk"][:150], "...\n")


if __name__ == "__main__":
    retriever = RbcRetriever()
    retriever.pretty_print("How do I report a lost credit card?")
