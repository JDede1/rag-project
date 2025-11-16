"""
split_compound_faqs.py
-------------------------------------------------------
Segment multi-scenario RBC FAQ entries into atomic Q/A units.

Goals:
    - Split long answers into focused segments
    - Use newline structure, anchor heuristics, and sentence boundaries
    - Preserve provenance (url, source, retrieved_at)
    - Produce clean atomic FAQ rows suitable for chunking

Input:
    rbc_faqs_normalized.parquet

Output:
    rbc_faqs_refined.parquet
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

INPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_normalized.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_refined.parquet"


# -------------------------------------------------------
# TEXT UTILS
# -------------------------------------------------------
def collapse_spaces(text: str) -> str:
    """
    Collapse excessive spaces on lines but preserve newline structure.
    """
    if not isinstance(text, str):
        return ""

    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Sentence splitter that preserves structure.
    Uses newline boundaries as soft segment boundaries.
    """
    if not isinstance(text, str):
        return []

    text = collapse_spaces(text)

    # Break by newline first (paragraph-level hints)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    sentences: List[str] = []
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"

    for para in paragraphs:
        parts = re.split(pattern, para)
        for p in parts:
            p = p.strip()
            if len(p) > 2:
                sentences.append(p)

    return sentences


# -------------------------------------------------------
# ANCHOR / SEGMENT LOGIC
# -------------------------------------------------------
ANCHOR_PREFIXES = (
    "if ",
    "when ",
    "whenever ",
    "in case ",
    "to ",
    "you can also ",
    "you may also ",
    "alternatively",
    "another option",
    "additionally",
    "or you can",
    "also, you can",
)


def looks_like_anchor(sentence: str) -> bool:
    """
    Identify sentences that begin new scenarios.
    """
    s = sentence.lower().strip()

    # Numbered or bullet-like
    if re.match(r"^(\d+[\).\s]+|-|\*)\s*", s):
        return True

    # Explicit scenario markers
    for prefix in ANCHOR_PREFIXES:
        if s.startswith(prefix):
            return True

    # Long "or"/"alternatively" logic
    if (" or " in s or " alternatively " in s) and len(s) > 60:
        return True

    return False


def get_anchor_indices(sentences: List[str]) -> List[int]:
    return [i for i, s in enumerate(sentences) if looks_like_anchor(s)]


def group_sentences_by_anchor(sentences: List[str], min_chars: int = 80) -> List[str]:
    """
    Group sentences into scenario-based segments.
    """
    if not sentences:
        return []

    anchors = get_anchor_indices(sentences)

    # No anchors: treat entire answer as a single segment
    if not anchors:
        combined = " ".join(sentences).strip()
        return [combined] if combined else []

    segments: List[str] = []

    for idx, anchor in enumerate(anchors):
        start = anchor
        end = anchors[idx + 1] if idx + 1 < len(anchors) else len(sentences)
        span = sentences[start:end]
        seg_text = " ".join(span).strip()
        if len(seg_text) >= min_chars:
            segments.append(seg_text)

    # Fallback if anchors yield too-small segments
    if not segments:
        combined = " ".join(sentences).strip()
        return [combined]

    return segments


# -------------------------------------------------------
# ATOMIC FAQ EXTRACTION
# -------------------------------------------------------
def extract_atomic_faqs(question: str, answer: str) -> List[Dict[str, str]]:
    """
    Produce atomic Q/A units from one FAQ row.
    """
    q = collapse_spaces(question)
    a = collapse_spaces(answer)

    if len(a) < 140:
        return [{"question": q, "answer": a}]

    sentences = split_into_sentences(a)

    if len(sentences) <= 2:
        return [{"question": q, "answer": a}]

    segments = group_sentences_by_anchor(sentences)

    atomic_items = []
    for seg in segments:
        seg_clean = seg.strip()
        if seg_clean:
            atomic_items.append({"question": q, "answer": seg_clean})

    if not atomic_items:
        atomic_items = [{"question": q, "answer": a}]

    return atomic_items


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def refine_faqs():
    print(f"Loading normalized FAQs from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} normalized FAQ rows")

    # Provenance fields
    provenance_cols = [
        col for col in ["url", "source", "retrieved_at"]
        if col in df.columns
    ]

    refined_rows: List[Dict] = []

    for idx, row in df.iterrows():
        atomic_list = extract_atomic_faqs(row["question"], row["answer"])

        for atomic_idx, item in enumerate(atomic_list):
            entry = {
                "question": item["question"],
                "answer": item["answer"],
                "source_faq_index": int(idx),
                "atomic_index": int(atomic_idx),
            }

            for col in provenance_cols:
                entry[col] = row[col]

            refined_rows.append(entry)

    refined_df = pd.DataFrame(refined_rows)

    # Clean and dedupe
    refined_df["question"] = refined_df["question"].astype(str).str.strip()
    refined_df["answer"] = refined_df["answer"].astype(str).str.strip()

    before = len(refined_df)
    refined_df = refined_df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    after = len(refined_df)

    print(f"Refined to {after} atomic FAQ entries (removed {before - after} duplicates).")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    refined_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved refined dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    refine_faqs()
