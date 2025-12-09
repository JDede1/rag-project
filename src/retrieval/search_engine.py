"""
Hybrid Search Engine — MPNet (Local) and ONNX (Cloud)
---------------------------------------------------------------------------

Modes:

    • Local (default)
        - SentenceTransformer MPNet encoder
        - Uses HuggingFace model (768-dim)

    • Cloud Run (production)
        - ONNX Runtime MPNet encoder (same 768-dim)
        - Loads ONNX + tokenizer from /app/data/index

Mode selection:

    DEPLOY_ENV=cloud → Cloud Run ONNX mode
    Default          → Local HF MPNet mode
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


# ---------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------
def _tokenize(text: str):
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------
# Topic classification rules
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
# Hybrid Retriever
# =========================================================
class RbcRetriever:
    def __init__(self):

        # Local = repo root
        # Cloud = /app
        if IS_CLOUD:
            base_dir = Path("/app")
        else:
            base_dir = Path(__file__).resolve().parents[2]

        index_dir = base_dir / "data" / "index"

        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.meta_path}")

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)

        self.faiss_dim = self.index.d
        print(f"[Retriever] Loaded FAISS index ({self.index.ntotal} vectors, dim={self.faiss_dim})")

        if IS_CLOUD:
            print("[Retriever] Cloud mode → ONNX MPNet")
            self._load_onnx_encoder()
        else:
            print("[Retriever] Local mode → SentenceTransformer MPNet")
            self._load_local_encoder()

    # =========================================================
    # LOCAL MPNet encoder
    # =========================================================
    def _load_local_encoder(self):

        if SentenceTransformer is None:
            raise RuntimeError("SentenceTransformer missing. Install sentence-transformers + torch")

        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.model = SentenceTransformer(model_name)

        # Validate embedding dimension
        test_emb = self.model.encode(["test"], convert_to_numpy=True)
        if test_emb.shape[1] != self.faiss_dim:
            raise ValueError(
                f"Local MPNet dim {test_emb.shape[1]} != FAISS dim {self.faiss_dim}\n"
                "You MUST keep MPNet encoder since your FAISS index was built with MPNet."
            )

        self.encoder_type = "mpnet_local"
        print("[Retriever] Local MPNet loaded.")

    # =========================================================
    # CLOUD: ONNX MPNet encoder
    # =========================================================
    def _load_onnx_encoder(self):
        try:
            import onnxruntime as ort
        except Exception:
            raise RuntimeError("Missing onnxruntime. Add it to requirements.txt")

        if IS_CLOUD:
            base_dir = Path("/app")
        else:
            base_dir = Path(__file__).resolve().parents[2]

        onnx_dir = base_dir / "data" / "index"

        self.onnx_path = onnx_dir / "mpnet.onnx"
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model missing: {self.onnx_path}")

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(onnx_dir),
            local_files_only=True
        )

        self.ort_session = ort.InferenceSession(str(self.onnx_path))

        dummy = self._encode_onnx_embeddings("test")
        if dummy.shape[1] != self.faiss_dim:
            raise ValueError(
                f"ERROR: ONNX MPNet dim {dummy.shape[1]} != FAISS index dim {self.faiss_dim}"
            )

        self.encoder_type = "mpnet_onnx"
        print("[Retriever] ONNX MPNet loaded.")

    # =========================================================
    # ONNX inference with mean pooling
    # =========================================================
    def _encode_onnx_embeddings(self, text: str) -> np.ndarray:

        tokens = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=256
        )

        ort_out = self.ort_session.run(None, tokens)[0]

        # Mean pool over sequence length
        if ort_out.ndim == 3:
            emb = ort_out.mean(axis=1)
        else:
            emb = ort_out

        if emb.ndim == 1:
            emb = np.expand_dims(emb, 0)

        emb = emb.astype(np.float32)
        faiss.normalize_L2(emb)
        return emb

    # =========================================================
    # EMBEDDING FUNCTION
    # =========================================================
    def embed_query(self, text: str):
        if self.encoder_type == "mpnet_onnx":
            return self._encode_onnx_embeddings(text)

        emb = self.model.encode([text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(emb)
        return emb

    # ---------------------------------------------------------
    # Reranking logic
    # ---------------------------------------------------------
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

    # =========================================================
    # MAIN SEARCH
    # =========================================================
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

    # =========================================================
    # Pretty printer
    # =========================================================
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
