"""
generate_embeddings.py
-------------------------------------
Generate vector embeddings for RBC FAQ dataset using Sentence Transformers.

Purpose:
    • Encode FAQ text (question + answer) into dense vector representations
    • Save embeddings and metadata for FAISS indexing
    • Enable future retrieval via semantic similarity

Usage:
    python src/embeddings/generate_embeddings.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_INDEX = BASE_DIR / "data" / "index"

DATA_INDEX.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def generate_embeddings():
    print("🔹 Loading cleaned RBC FAQ dataset...")
    df = pd.read_parquet(DATA_PROCESSED / "rbc_faqs.parquet")
    print(f"✅ Loaded {len(df)} records")

    # Combine question and answer into one semantic unit
    df["combined"] = df["question"].str.strip() + " " + df["answer"].str.strip()

    # Load model
    print(f"🧠 Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Encode
    print("⚙️ Generating embeddings...")
    embeddings = model.encode(
        df["combined"].tolist(),
        show_progress_bar=True,
        batch_size=16,
        convert_to_numpy=True
    )

    print(f"✅ Embeddings shape: {embeddings.shape}")

    # Save artifacts
    np.save(DATA_INDEX / "rbc_embeddings.npy", embeddings)
    df[["question", "answer", "url"]].to_parquet(DATA_INDEX / "rbc_metadata.parquet", index=False)

    print(f"💾 Saved embeddings → {DATA_INDEX / 'rbc_embeddings.npy'}")
    print(f"💾 Saved metadata → {DATA_INDEX / 'rbc_metadata.parquet'}")
    print("✨ Embedding generation complete.")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    generate_embeddings()
