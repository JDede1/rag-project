"""
generate_embeddings.py
-------------------------------------------------------
Embedding Generator (MiniLM Upgrade)

Purpose:
    • Generate embeddings using all-MiniLM-L6-v2 (384-dim)
    • Uses CHUNK-ONLY text + light category hint
    • Prevents topic collisions ('lost card' ↔ 'fraud')
    • Produces Cloud-Run–compatible embeddings
    • Saves:
          - rbc_embeddings.npy
          - rbc_metadata.parquet

This script is executed LOCALLY or in COLAB (Phase 3),
NOT inside Cloud Run.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


# -------------------------------------------------------
# CLOUD / LOCAL PATH RESOLUTION (unified)
# -------------------------------------------------------
IS_CLOUD = os.getenv("DEPLOY_ENV", "").lower() == "cloud"

if IS_CLOUD:
    # Cloud Run always mounts repo to /app
    PROJECT_ROOT = Path("/app")
else:
    # Local/Colab: go 2 directories up from this file
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INDEX = PROJECT_ROOT / "data" / "index"
DATA_INDEX.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# MODEL: MiniLM (Cloud-Run compatible)
# -------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = DATA_PROCESSED / "rbc_faq_chunks.parquet"


# -------------------------------------------------------
# LIGHT CATEGORY SIGNAL
# -------------------------------------------------------
def classify_hint(question: str) -> str:
    """
    Adds a tiny category signal to avoid semantic drift.
    Safe + proven in Phase 7.
    """
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
# MAIN: Generate Embeddings
# -------------------------------------------------------
def generate_embeddings():
    print("\n===============================================")
    print("   Generating MiniLM Embeddings (Phase 3)")
    print("===============================================\n")

    print(f"Loading chunked FAQ dataset: {CHUNKS_PATH}")
    df = pd.read_parquet(CHUNKS_PATH)
    print(f"Loaded {len(df)} chunks")

    # ---------------------------------------------------
    # Build embedding text
    # ---------------------------------------------------
    df["category_hint"] = df["question"].apply(classify_hint)

    df["embedding_text"] = (
        df["chunk"].str.strip() +
        df["category_hint"]
    )

    print("\nSample embedding_text:")
    print(df["embedding_text"].iloc[0][:200], "...\n")

    # ---------------------------------------------------
    # Load MiniLM model
    # ---------------------------------------------------
    print(f"Loading MiniLM embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------
    print("Generating embeddings... (this may take a while)\n")

    embeddings = model.encode(
        df["embedding_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        device="cuda" if model.device is not None else None
    )

    print(f"\nEmbeddings shape (expect 384 dim): {embeddings.shape}\n")

    # ---------------------------------------------------
    # Save embeddings
    # ---------------------------------------------------
    emb_path = DATA_INDEX / "rbc_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings → {emb_path}")

    # ---------------------------------------------------
    # Save metadata
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

    print(f"✓ Saved metadata → {meta_path}")
    print("\nEmbedding generation complete.\n")


# -------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------
if __name__ == "__main__":
    generate_embeddings()
