"""
chunk_text.py
-------------------------------------------------------
Chunking module for RAG preprocessing.

Purpose:
    Convert refined FAQ entries into retrieval-friendly chunks suitable
    for embedding and FAISS indexing.

Enhancements in this version:
    • Preserves provenance fields:
        - source_faq_index
        - url
        - source
        - retrieved_at
    • Ensures metadata flows into embedding + FAISS layers
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
def build_chunks_for_faq(question: str,
                         answer: str,
                         provenance: Dict,
                         max_chars: int = 600,
                         min_chars: int = 150) -> List[Dict]:
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

    chunks = []
    current = ""

    for sent in sentences:
        if not current:
            current = sent
            continue

        if len(current) + len(sent) + 1 <= max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent

    if current:
        chunks.append(current.strip())

    # Merge small chunks
    merged = []
    buffer = ""

    for chunk in chunks:
        if len(chunk) < min_chars:
            buffer += " " + chunk
            continue

        if buffer.strip():
            merged.append(buffer.strip())
            buffer = ""

        merged.append(chunk.strip())

    if buffer.strip():
        merged.append(buffer.strip())

    # Build final chunk entries (with provenance)
    output = []
    for chunk in merged:
        entry = {
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

    all_chunks = []

    for _, row in df.iterrows():
        q = row["question"]
        a = row["answer"]

        # Collect provenance metadata for this FAQ row
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
