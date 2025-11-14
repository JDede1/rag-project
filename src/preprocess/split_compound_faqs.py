"""
split_compound_faqs.py
-------------------------------------------------------
Enterprise-grade splitter for compound RBC FAQ entries.

Goals:
    • Turn one long, multi-scenario FAQ into several
      smaller, atomic Q/A units.
    • Preserve provenance:
        - url
        - source
        - retrieved_at
        - source_faq_index (row index in normalized file)
    • Use semantic-ish heuristics instead of only
      "question-looking" lines.
    • Greatly improve RAG retrieval quality by:
        - Shorter, focused answers
        - Clear scenario boundaries
        - Less noise per atomic FAQ

Pipeline:
    1. Load rbc_faqs_normalized.parquet
    2. For each row:
        - Split answer into sentences
        - Detect "anchor" sentences (If / When / You can also / etc.)
        - Group sentences into coherent segments
    3. Produce atomic FAQ rows with:
        question, answer, source_faq_index, url, source, retrieved_at

Usage:
    python src/preprocess/split_compound_faqs.py
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
# TEXT UTILITIES
# -------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # replace multiple spaces/newlines with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter:
        - splits on ., ?, !
        - keeps abbreviations relatively safe (best-effort)
    We prefer a robust heuristic over heavy external deps (spaCy, nltk).
    """
    text = normalize_whitespace(text)

    # Protect common abbreviations by temporary token
    protect_map = {
        "e.g.": "e<abbr>g<dot>",
        "i.e.": "i<abbr>e<dot>",
        "etc.": "etc<dot>",
    }
    for k, v in protect_map.items():
        text = text.replace(k, v)

    # Split on sentence-ending punctuation
    parts = re.split(r"([.!?])\s+", text)

    sentences = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        if part in [".", "!", "?"]:
            buffer += part
            sentences.append(buffer.strip())
            buffer = ""
        else:
            if buffer:
                buffer += " " + part
            else:
                buffer = part

    if buffer:
        sentences.append(buffer.strip())

    # Restore abbreviations
    restored = []
    for s in sentences:
        for k, v in protect_map.items():
            s = s.replace(v, k)
        restored.append(s.strip())

    # Filter extremely short garbage
    return [s for s in restored if len(s) > 2]


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
    "or you can",
    "additionally",
    "also, you can",
)


def looks_like_anchor(sentence: str) -> bool:
    """
    Returns True if the sentence looks like the start
    of a distinct scenario / sub-procedure.
    """
    s = sentence.strip().lower()

    # numbered / bulleted
    if re.match(r"^[-•\d]+\s", s):
        return True

    # starts with any of our scenario markers
    for prefix in ANCHOR_PREFIXES:
        if s.startswith(prefix):
            return True

    # fallback: long sentence that contains 'or' / 'alternatively'
    if (" or " in s or " alternatively " in s) and len(s) > 60:
        return True

    return False


def get_anchor_indices(sentences: List[str]) -> List[int]:
    """
    Return indices of sentences that start new mini-scenarios.
    """
    anchors = []
    for i, sent in enumerate(sentences):
        if looks_like_anchor(sent):
            anchors.append(i)
    return anchors


def group_sentences_by_anchor(
    sentences: List[str],
    min_answer_chars: int = 60,
) -> List[str]:
    """
    Group sentences into semantic-ish segments using anchor
    sentences as boundaries.

    Rules:
        • If no anchors → single segment = whole answer.
        • If anchors present:
            - each anchor starts a new segment
            - segment ends at next anchor or end of list
            - drop segments that are too tiny
    """
    if not sentences:
        return []

    anchors = get_anchor_indices(sentences)

    # No anchors → just return entire answer as one segment
    if len(anchors) <= 1:
        joined = " ".join(sentences).strip()
        return [joined] if len(joined) >= min_answer_chars else [joined]

    segments: List[str] = []
    for idx, start in enumerate(anchors):
        end = anchors[idx + 1] if idx + 1 < len(anchors) else len(sentences)
        span_sents = sentences[start:end]
        seg_text = " ".join(span_sents).strip()

        if len(seg_text) >= min_answer_chars:
            segments.append(seg_text)

    # Fallback: if all segments are too small, fall back to full answer
    if not segments:
        full = " ".join(sentences).strip()
        return [full]

    return segments


# -------------------------------------------------------
# ATOMIC FAQ EXTRACTION
# -------------------------------------------------------
def extract_atomic_faqs(question: str, answer: str) -> List[Dict[str, str]]:
    """
    Split a single normalized FAQ into atomic Q/A pairs.

    Strategy:
        1. Normalize answer & split into sentences
        2. Group by semantic anchors
        3. Each segment becomes an "atomic answer"
        4. Question text is kept as the original question
           (we avoid noisy auto-rewriting)
    """
    q = normalize_whitespace(question)
    a = normalize_whitespace(answer)

    # Very short answers → no splitting
    if len(a) < 120:
        return [{"question": q, "answer": a}]

    sentences = split_into_sentences(a)

    # still low granularity → no further work
    if len(sentences) <= 2:
        return [{"question": q, "answer": a}]

    segments = group_sentences_by_anchor(sentences)

    atomic_list: List[Dict[str, str]] = []
    for seg in segments:
        seg_norm = seg.strip()
        if not seg_norm:
            continue
        atomic_list.append(
            {
                "question": q,
                "answer": seg_norm,
            }
        )

    # Fallback if we somehow produced nothing
    if not atomic_list:
        atomic_list = [{"question": q, "answer": a}]

    return atomic_list


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def refine_faqs():
    print(f"🔹 Loading normalized FAQs from: {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} normalized FAQ rows")

    # Determine which provenance columns exist
    provenance_cols = [c for c in ["url", "source", "retrieved_at"] if c in df.columns]

    refined_rows: List[Dict] = []

    for idx, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]

        atomic_items = extract_atomic_faqs(question, answer)

        for atomic_idx, item in enumerate(atomic_items):
            entry = {
                "question": item["question"],
                "answer": item["answer"],
                "source_faq_index": int(idx),
                "atomic_index": int(atomic_idx),
            }

            # propagate provenance (if present)
            for col in provenance_cols:
                entry[col] = row[col]

            refined_rows.append(entry)

    refined_df = pd.DataFrame(refined_rows)

    # Final cleanup: trim whitespace, drop exact dupes
    refined_df["question"] = refined_df["question"].astype(str).str.strip()
    refined_df["answer"] = refined_df["answer"].astype(str).str.strip()

    before = len(refined_df)
    refined_df = refined_df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    after = len(refined_df)

    print(f"Refined to {after} atomic FAQ entries (dropped {before - after} duplicates).")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    refined_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved refined dataset → {OUTPUT_PATH}")


if __name__ == "__main__":
    refine_faqs()
