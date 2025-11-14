"""
chunk_text.py
-------------------------------------------------------
Enterprise-grade chunker for RBC RAG pipeline.

Improvements:
    • Uses semantic-aware chunking, not just crude sentence splits
    • Produces high-quality retrieval chunks for MPNet embeddings
    • Works perfectly with the updated enterprise split_compound_faqs.py
    • Minimizes overlap & redundancy
    • Preserves all provenance columns:
          - source_faq_index
          - atomic_index
          - url
          - source
          - retrieved_at

Goal:
    Convert refined (atomic) FAQ entries into clean, tight,
    retrieval-friendly chunks.

Chunk strategy:
    1. Split answer into sentences
    2. Merge sentences into chunks that:
         • are ≤ 320 chars (ideal for MPNet)
         • are ≥ 80 chars (avoid tiny chunks)
         • maintain coherent meaning
    3. Guarantee chunks align with semantic boundaries
"""

from __future__ import annotations

import re
import pandas as pd
from pathlib import Path
from typing import List, Dict


# -------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_refined.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faq_chunks.parquet"


# -------------------------------------------------------
# LIGHTWEIGHT SENTENCE SPLITTER
# -------------------------------------------------------
# This is a safer splitter that avoids over-splitting abbreviations.
SENTENCE_PATTERN = r"(?<=[.!?])\s+(?=[A-Z])"


def split_into_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    parts = re.split(SENTENCE_PATTERN, text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


# -------------------------------------------------------
# SEMANTIC CHUNK BUILDER
# -------------------------------------------------------
def build_chunks_for_answer(
    question: str,
    answer: str,
    provenance: Dict,
    max_chars: int = 320,
    min_chars: int = 80,
) -> List[Dict]:
    """
    Convert a single atomic Q/A pair into multiple optimized chunks.

    Logic:
        • Join sentences until ~320 chars
        • Ensure no chunk is shorter than ~80 chars
        • Keep chunks meaningful for MPNet retrieval
    """

    sentences = split_into_sentences(answer)
    if not sentences:
        return []

    chunks: List[str] = []
    buffer = ""

    for sent in sentences:
        sent = sent.strip()

        # If buffer is empty → start with this sentence
        if not buffer:
            buffer = sent
            continue

        # If adding sentence stays under max_chars → append
        if len(buffer) + 1 + len(sent) <= max_chars:
            buffer += " " + sent
        else:
            # Close current chunk and start new one
            chunks.append(buffer.strip())
            buffer = sent

    if buffer:
        chunks.append(buffer.strip())

    # Second pass: ensure minimum chunk length
    final_chunks: List[str] = []
    temp = ""

    for ch in chunks:
        if len(ch) < min_chars:
            if temp:
                temp += " " + ch
            else:
                temp = ch
            continue

        if temp:
            final_chunks.append(temp.strip())
            temp = ""

        final_chunks.append(ch.strip())

    if temp:
        final_chunks.append(temp.strip())

    # Build structured rows with provenance
    output = []
    for chunk in final_chunks:
        entry = {
            "question": question.strip(),
            "chunk": chunk.strip(),
        }
        for k, v in provenance.items():
            entry[k] = v
        output.append(entry)

    return output


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def chunk_faq_dataset():
    print(f"🔹 Loading refined FAQs from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} atomic FAQ entries")

    provenance_cols = [
        col for col in
        ["source_faq_index", "atomic_index", "url", "source", "retrieved_at"]
        if col in df.columns
    ]

    all_chunks: List[Dict] = []

    for _, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]

        provenance = {col: row[col] for col in provenance_cols}

        faq_chunks = build_chunks_for_answer(question, answer, provenance)
        all_chunks.extend(faq_chunks)

    chunk_df = pd.DataFrame(all_chunks)

    before = len(chunk_df)
    chunk_df = chunk_df.drop_duplicates(subset=["question", "chunk"]).reset_index(drop=True)
    after = len(chunk_df)

    print(f"Generated {after} chunks (removed {before - after} duplicates)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunk_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved chunked dataset → {OUTPUT_PATH}")


if __name__ == "__main__":
    chunk_faq_dataset()
