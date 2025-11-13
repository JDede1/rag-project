"""
generate_embeddings.py
-------------------------------------------------------
Generate vector embeddings for RBC FAQ chunks using
Sentence Transformers, while preserving provenance metadata.

This version supports the upgraded preprocessing pipeline:
    • clean → normalize → split → chunk
    • final dataset: rbc_faq_chunks.parquet

Each chunk contains:
    question
    chunk
    source_faq_index
    url
    source
    retrieved_at

Outputs:
    • rbc_embeddings.npy
    • rbc_metadata.parquet
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_INDEX = BASE_DIR / "data" / "index"

DATA_INDEX.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = DATA_PROCESSED / "rbc_faq_chunks.parquet"


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
def generate_embeddings():
    print(f"Loading chunked FAQ dataset: {CHUNKS_PATH.name}")
    df = pd.read_parquet(CHUNKS_PATH)
    print(f"Loaded {len(df)} chunk entries")

    # ---------------------------------------------------
    # Build embedding text
    # ---------------------------------------------------
    # Format: "Question? Chunk text..."
    df["embedding_text"] = df["question"].str.strip() + " " + df["chunk"].str.strip()

    # ---------------------------------------------------
    # Load SentenceTransformer model
    # ---------------------------------------------------
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------
    # Generate embeddings
    # ---------------------------------------------------
    print("Generating embeddings...")
    embeddings = model.encode(
        df["embedding_text"].tolist(),
        show_progress_bar=True,
        batch_size=16,
        convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # ---------------------------------------------------
    # Save embeddings
    # ---------------------------------------------------
    emb_path = DATA_INDEX / "rbc_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings → {emb_path}")

    # ---------------------------------------------------
    # Save metadata (without embedding_text)
    # ---------------------------------------------------
    metadata_cols = [
        "question",
        "chunk",
        "source_faq_index",
        "url",
        "source",
        "retrieved_at",
    ]

    # Only keep columns that appear in the dataset
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
