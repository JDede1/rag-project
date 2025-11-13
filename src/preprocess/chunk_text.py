"""
chunk_text.py
-------------------------------------------------------
Chunking module for RAG preprocessing.

Purpose:
    Convert cleaned and refined FAQ entries into retrieval-friendly
    chunks suitable for embedding and FAISS indexing.

Design requirements:
    • Preserve question → answer relationship
    • Split long answers into manageable segments
    • Use sentence boundaries to avoid mid-sentence breaks
    • Maintain a maximum token/character size per chunk
    • Avoid producing excessively small or meaningless fragments
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
def build_chunks_for_faq(question: str, answer: str,
                         max_chars: int = 600,
                         min_chars: int = 150) -> List[Dict]:
    """
    Chunk a single Q/A pair using sentence-based grouping.

    Parameters:
        question (str): Source FAQ question
        answer (str): Cleaned answer text
        max_chars (int): Maximum characters per chunk
        min_chars (int): Merge smaller sentence groups to reach this length

    Returns:
        List[Dict]: List of chunk records
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

    # Merge chunks that are too small
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

    # Build output structure
    return [
        {
            "question": question.strip(),
            "chunk": chunk.strip()
        }
        for chunk in merged
        if chunk.strip()
    ]


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def chunk_faq_dataset():
    print(f"Loading {INPUT_PATH.name}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} FAQ entries")

    all_chunks = []

    for _, row in df.iterrows():
        q = row["question"]
        a = row["answer"]

        faq_chunks = build_chunks_for_faq(q, a)
        all_chunks.extend(faq_chunks)

    out_df = pd.DataFrame(all_chunks).drop_duplicates(subset=["question", "chunk"])

    print(f"Generated {len(out_df)} chunks")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved chunked dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    chunk_faq_dataset()
