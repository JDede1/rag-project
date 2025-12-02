"""
generate_embeddings.py
-------------------------------------------------------
Embedding Generator

Fixes the major retrieval issue where:
    • 'lost card' queries retrieved 'fraud' chunks
    • 'fraud' queries retrieved 'lost card' chunks
    • Retrieval relied too heavily on question text

New strategy:
    • Embed CHUNK-ONLY text (removes semantic collision)
    • Add an extremely light category signal (optional, safe)
    • Preserve all provenance and file paths

Outputs:
    • rbc_embeddings.npy
    • rbc_metadata.parquet
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_INDEX = BASE_DIR / "data" / "index"

DATA_INDEX.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNKS_PATH = DATA_PROCESSED / "rbc_faq_chunks.parquet"


# -------------------------------------------------------
# LIGHT CATEGORY SIGNAL
# -------------------------------------------------------
# This prevents semantic drift without reintroducing collision.
def classify_hint(question: str) -> str:
    q = question.lower()

    if any(k in q for k in ["lost", "stolen", "misplaced"]):
        return " lostcard"
    if any(k in q for k in ["fraud", "unauthorized", "dispute"]):
        return " fraud"
    if any(k in q for k in ["password", "login", "reset"]):
        return " login"
    if any(k in q for k in ["transfer", "etransfer", "e-transfer", "interac"]):
        return " etransfer"
    return " general"


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
def generate_embeddings():
    print(f"Loading chunked FAQ dataset: {CHUNKS_PATH.name}")
    df = pd.read_parquet(CHUNKS_PATH)
    print(f"Loaded {len(df)} chunk entries")

    # ---------------------------------------------------
    # Build text to embed (CHUNK-ONLY + category hint)
    # ---------------------------------------------------
    # Major Phase-7 fix: avoid question+chunk collisions
    df["category_hint"] = df["question"].apply(classify_hint)

    df["embedding_text"] = (
        df["chunk"].str.strip()
        + df["category_hint"]         # tiny nudge, keeps clusters distinct
    )

    print("Sample embedding_text:", df["embedding_text"].iloc[0][:120], "...")

    # ---------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------
    # Generate embeddings (GPU-optimized)
    # ---------------------------------------------------
    print("Generating embeddings...")

    embeddings = model.encode(
        df["embedding_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        device="cuda" if model.device is not None else None
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # ---------------------------------------------------
    # Save embeddings → FAISS will normalize them
    # ---------------------------------------------------
    emb_path = DATA_INDEX / "rbc_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings → {emb_path}")

    # ---------------------------------------------------
    # Save metadata (unchanged)
    # ---------------------------------------------------
    metadata_cols = [
        "question",
        "chunk",
        "source_faq_index",
        "url",
        "source",
        "retrieved_at",
    ]

    metadata_cols = [c for c in metadata_cols if c in df.columns]

    metadata = df[metadata_cols].copy()

    meta_path = DATA_INDEX / "rbc_metadata.parquet"
    metadata.to_parquet(meta_path, index=False)

    print(f"Saved metadata → {meta_path}")
    print("Embedding generation complete.")


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    generate_embeddings()
