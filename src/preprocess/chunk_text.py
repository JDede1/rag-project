"""
chunk_text.py
-------------------------------------------------------
Chunking module for RAG preprocessing.

Purpose:
    Convert refined FAQ entries into retrieval-friendly chunks suitable
    for embedding and FAISS indexing.

This version:
    • Uses smaller, tighter chunks optimized for mpnet embeddings
    • Preserves provenance fields:
        - source_faq_index
        - url
        - source
        - retrieved_at
"""

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
# SENTENCE SPLITTING
# -------------------------------------------------------
SENTENCE_REGEX = r"(?<=[.!?])\s+(?=[A-Z])"


def split_into_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = re.split(SENTENCE_REGEX, text.strip())
    return [s.strip() for s in sentences if s.strip()]


# -------------------------------------------------------
# CHUNK BUILDING LOGIC
# -------------------------------------------------------
def build_chunks_for_faq(
    question: str,
    answer: str,
    provenance: Dict,
    max_chars: int = 300,
    min_chars: int = 80,
) -> List[Dict]:
    """
    Chunk a single Q/A pair using sentence-based grouping.

    Parameters:
        question (str): Source FAQ question
        answer (str): Cleaned answer text
        provenance (dict): Metadata fields to propagate into each chunk
        max_chars (int): Maximum characters per chunk
        min_chars (int): Merge smaller sentence groups to reach this length

    Returns:
        List[Dict]: List of chunk records with provenance
    """

    sentences = split_into_sentences(answer)
    if not sentences:
        return []

    # First pass: group sentences into preliminary chunks under max_chars
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if not current:
            current = sent
            continue

        # If adding this sentence stays within max_chars, append
        if len(current) + 1 + len(sent) <= max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    # Second pass: merge very short chunks into neighbors
    merged: List[str] = []
    buffer = ""

    for chunk in chunks:
        if len(chunk) < min_chars:
            # Accumulate short fragments
            if buffer:
                buffer += " " + chunk
            else:
                buffer = chunk
            continue

        # If there is a buffered short chunk, flush it first
        if buffer:
            merged.append(buffer.strip())
            buffer = ""

        merged.append(chunk.strip())

    if buffer:
        merged.append(buffer.strip())

    # Build final chunk entries (with provenance)
    output: List[Dict] = []
    for chunk in merged:
        entry: Dict = {
            "question": question.strip(),
            "chunk": chunk.strip(),
        }
        # Attach provenance metadata
        for key, value in provenance.items():
            entry[key] = value

        output.append(entry)

    return output


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def chunk_faq_dataset():
    print(f"Loading {INPUT_PATH.name}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} FAQ entries")

    # Identify provenance columns
    provenance_cols = []
    for col in ["source_faq_index", "url", "source", "retrieved_at"]:
        if col in df.columns:
            provenance_cols.append(col)

    all_chunks: List[Dict] = []

    for _, row in df.iterrows():
        q = row["question"]
        a = row["answer"]

        provenance = {col: row[col] for col in provenance_cols}
        faq_chunks = build_chunks_for_faq(q, a, provenance)
        all_chunks.extend(faq_chunks)

    out_df = pd.DataFrame(all_chunks).drop_duplicates(subset=["question", "chunk"])

    print(f"Generated {len(out_df)} chunks")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved chunked dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    chunk_faq_dataset()
