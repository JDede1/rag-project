"""
chunk_text.py
-------------------------------------------------------
Chunking module for RAG preprocessing.

Purpose:
    Convert refined atomic FAQ entries into retrieval-friendly chunks
    suitable for MPNet embeddings and FAISS indexing.

This version:
    • Delegates sentence/segment logic to semantic_segmenter.py
    • Produces tight, coherent chunks in the 80–320 character range
    • Preserves provenance fields:
          - source_faq_index
          - atomic_index (if present)
          - url
          - source
          - retrieved_at
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.preprocess.semantic_segmenter import segment_answer_into_chunks


# -------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_refined.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faq_chunks.parquet"


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def chunk_faq_dataset():
    print(f"Loading refined FAQs from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} atomic FAQ rows")

    provenance_cols = [
        col for col in ["source_faq_index", "atomic_index", "url", "source", "retrieved_at"]
        if col in df.columns
    ]

    all_chunks: List[Dict] = []

    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        if not answer:
            continue

        provenance = {col: row[col] for col in provenance_cols}

        chunk_texts = segment_answer_into_chunks(
            answer=answer,
            max_chars=320,
            min_chars=80,
        )

        for chunk in chunk_texts:
            entry: Dict = {
                "question": question,
                "chunk": chunk.strip(),
            }
            for k, v in provenance.items():
                entry[k] = v

            all_chunks.append(entry)

    if not all_chunks:
        raise RuntimeError("No chunks were generated. Check input data and configuration.")

    chunk_df = pd.DataFrame(all_chunks)

    before = len(chunk_df)
    chunk_df = chunk_df.drop_duplicates(subset=["question", "chunk"]).reset_index(drop=True)
    after = len(chunk_df)

    print(f"Generated {after} unique chunks (removed {before - after} duplicates)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunk_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved chunked dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    chunk_faq_dataset()
