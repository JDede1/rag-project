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
    DEPLOY_ENV=local → Local mode

Optional override (local only):

    ENCODER_MODE = minilm_local_onnx
    ENCODER_MODE = minilm_cloud_onnx
    ENCODER_MODE = minilm_hf
"""

import os
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# ENVIRONMENT MODE
# ---------------------------------------------------------
IS_CLOUD = os.getenv("DEPLOY_ENV", "").lower() == "cloud"
ENCODER_OVERRIDE = os.getenv("ENCODER_MODE", "").strip().lower()

# sentence-transformers available only locally
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

        # root folder for index
        if IS_CLOUD:
            base_dir = Path("/app")
        else:
            base_dir = Path(__file__).resolve().parents[2]

        index_dir = base_dir / "data" / "index"

        # core index files
        self.index_path = index_dir / "rbc_faiss.index"
        self.meta_path = index_dir / "rbc_metadata.parquet"

        # ONNX variants
        self.onnx_cloud = index_dir / "minilm.onnx"           # IR ≤ 9
        self.onnx_local = index_dir / "minilm_local.onnx"     # IR 10 (local only)

        # verify data exists
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {self.meta_path}")

        # load index + metadata
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = pd.read_parquet(self.meta_path)
        self.faiss_dim = self.index.d

        print(
            f"[Retriever] Loaded FAISS index "
            f"({self.index.ntotal} vectors, dim={self.faiss_dim})"
        )

        # ============================
        # SELECT MODE
        # ============================

        if ENCODER_OVERRIDE:
            print(f"[Retriever] Manual override → {ENCODER_OVERRIDE}")
            self._apply_manual_override(ENCODER_OVERRIDE)
            return

        if IS_CLOUD:
            print("[Retriever] Cloud mode → minilm.onnx (Cloud-safe IR ≤ 9)")
            self._load_cloud_onnx()
        else:
            if self.onnx_local.exists():
                print("[Retriever] Local mode → minilm_local.onnx (IR 10)")
                self._load_local_onnx()
            else:
                print("[Retriever] Local mode → HF MiniLM")
                self._load_local_hf()

    # ==================================================================
    # MANUAL OVERRIDE (optional)
    # ==================================================================
    def _apply_manual_override(self, mode: str):
        mode = mode.lower()

        if mode == "minilm_local_onnx":
            self._load_local_onnx()
        elif mode == "minilm_cloud_onnx":
            self._load_cloud_onnx()
        elif mode == "minilm_hf":
            self._load_local_hf()
        else:
            raise ValueError(f"Unknown ENCODER_MODE override: {mode}")

    # ==================================================================
    # LOCAL HF MiniLM
    # ==================================================================
    def _load_local_hf(self):

        if SentenceTransformer is None:
            raise RuntimeError("SentenceTransformer unavailable.")

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name)

        # check dimension
        dim = self.model.encode(["test"], convert_to_numpy=True).shape[1]
        if dim != self.faiss_dim:
            raise ValueError(
                f"HF MiniLM dim {dim} != FAISS dim {self.faiss_dim} — rebuild FAISS."
            )

        self.encoder_type = "minilm_hf"
        print("[Retriever] Loaded HF MiniLM.")

    # ==================================================================
    # CLOUD ONNX (IR ≤ 9)
    # ==================================================================
    def _load_cloud_onnx(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not self.onnx_cloud.exists():
            raise FileNotFoundError(f"Cloud ONNX missing: {self.onnx_cloud}")

        onnx_dir = self.onnx_cloud.parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir), local_files_only=True)
        self.ort = ort.InferenceSession(str(self.onnx_cloud))

        self._check_onnx_dim()

        self.encoder_type = "minilm_cloud_onnx"
        print("[Retriever] Loaded Cloud ONNX MiniLM.")

    # ==================================================================
    # LOCAL ONNX (IR 10)
    # ==================================================================
    def _load_local_onnx(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not self.onnx_local.exists():
            raise FileNotFoundError(f"Local ONNX missing: {self.onnx_local}")

        onnx_dir = self.onnx_local.parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir), local_files_only=True)
        self.ort = ort.InferenceSession(str(self.onnx_local))

        self._check_onnx_dim()

        self.encoder_type = "minilm_local_onnx"
        print("[Retriever] Loaded Local ONNX MiniLM.")

    # ==================================================================
    # ONNX DIMENSION CHECK
    # ==================================================================
    def _check_onnx_dim(self):
        emb = self._encode_onnx("test")
        if emb.shape[1] != self.faiss_dim:
            raise ValueError(
                f"ONNX encoder dim {emb.shape[1]} != FAISS dim {self.faiss_dim}"
            )

    # ==================================================================
    # ONNX ENCODING
    # ==================================================================
    def _encode_onnx(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=256,
        )

        out = self.ort.run(None, tokens)[0]

        if out.ndim == 3:
            out = out.mean(axis=1)

        out = out.astype(np.float32)
        if out.ndim == 1:
            out = np.expand_dims(out, 0)

        faiss.normalize_L2(out)
        return out

    # ==================================================================
    # EMBEDDING ROUTER
    # ==================================================================
    def embed_query(self, text: str):
        if self.encoder_type in {"minilm_cloud_onnx", "minilm_local_onnx"}:
            return self._encode_onnx(text)

        # HF mode
        emb = self.model.encode([text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(emb)
        return emb

    # ==================================================================
    # RERANK
    # ==================================================================
    def _rerank(self, query, results):

        q_tokens = set(_tokenize(query))
        q_topic = classify_query_topic(query)
        has_q = bool(q_tokens & QUESTION_WORDS)

        reranked = []

        for i, r in enumerate(results):

            chunk = r["chunk"]
            chunk_topic = classify_chunk_topic(chunk)
            chunk_tokens = set(_tokenize(chunk))
            lexical = len(q_tokens & chunk_tokens)

            base = r["score"]
            lexical_boost = 0.12 * lexical
            q_boost = 0.05 if has_q else 0

            if q_topic == chunk_topic:
                topic_boost = 0.30
            elif q_topic != "general" and chunk_topic != "general":
                topic_boost = -0.10
            else:
                topic_boost = 0

            final = float(base + lexical_boost + q_boost + topic_boost)

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
