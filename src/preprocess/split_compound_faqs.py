"""
split_compound_faqs.py
-------------------------------------------------------
Splits compound FAQ entries into atomic Q/A units.

Problem:
    Scraped RBC FAQ pages sometimes merge multiple Q/A pairs into
    a single large answer block. This module identifies and splits
    those blocks safely without damaging valid answers.

Principles:
    • Never split unless structural cues are present
    • Preserve valid Q–A units
    • Handle bullet-style embedded questions
    • Avoid naive regex splitting that breaks sentences
"""

import re
import pandas as pd
from pathlib import Path


# -------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

# Updated to use normalized data (correct sequence: clean → normalize → split)
INPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_normalized.parquet"

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_refined.parquet"


# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------
QUESTION_START_PATTERNS = [
    r"^[A-Z].*\?$",                    # Full standalone question
    r"^-?\s*[A-Z].*?\?$",              # Bullet question: "- How do I apply?"
    r"^\d+\.\s*[A-Z].*?\?$",           # Numbered question: "1. What is my limit?"
]


def is_question_line(text: str) -> bool:
    """
    Determine whether a line is likely a standalone question.
    """
    stripped = text.strip()
    if not stripped:
        return False

    for pattern in QUESTION_START_PATTERNS:
        if re.match(pattern, stripped):
            return True

    return False


def extract_atomic_faqs(question: str, answer: str):
    """
    Identify sub-questions inside an answer and return
    a list of atomic {question, answer} pairs.

    Strategy:
        1. Split answer into lines
        2. Detect lines that are standalone questions
        3. Partition answer by these boundaries
        4. Assign corresponding answer blocks
    """
    lines = [l.strip() for l in answer.split("\n") if l.strip()]

    # Detect all lines that appear to be sub-questions
    question_indices = [i for i, line in enumerate(lines) if is_question_line(line)]

    # If no internal questions found, keep original
    if len(question_indices) <= 1:
        return [{"question": question.strip(), "answer": answer.strip()}]

    atomic_pairs = []

    for idx, q_index in enumerate(question_indices):
        sub_q = lines[q_index]

        # Determine answer boundaries
        start = q_index + 1
        end = question_indices[idx + 1] if idx + 1 < len(question_indices) else len(lines)

        sub_a_lines = lines[start:end]
        sub_a = " ".join(sub_a_lines).strip()

        if len(sub_a) < 10:
            continue

        # Normalize question text
        sub_q = re.sub(r"^[-\d\.\s]+", "", sub_q).strip()

        atomic_pairs.append({"question": sub_q, "answer": sub_a})

    if not atomic_pairs:
        return [{"question": question.strip(), "answer": answer.strip()}]

    return atomic_pairs


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def refine_faqs():
    print(f"Loading {INPUT_PATH.name}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    refined_rows = []

    for _, row in df.iterrows():
        q = row["question"]
        a = row["answer"]
        refined_rows.extend(extract_atomic_faqs(q, a))

    refined_df = pd.DataFrame(refined_rows).drop_duplicates(subset=["question", "answer"])

    refined_df["question"] = refined_df["question"].str.strip()
    refined_df["answer"] = refined_df["answer"].str.strip()

    print(f"Refined to {len(refined_df)} atomic FAQ entries")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    refined_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved refined dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    refine_faqs()
