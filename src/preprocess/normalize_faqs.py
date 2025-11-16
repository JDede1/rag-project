"""
normalize_faqs.py
-------------------------------------------------------
Normalize FAQ entries so that all question/answer pairs:

    - Follow consistent structure
    - Have valid questions
    - Preserve answer structure
    - Remove mislabeled headers or navigation elements
    - Prepare the dataset for splitting and chunking

This version preserves provenance fields:
    - url
    - source
    - retrieved_at

Runs AFTER clean_rbc_faqs.py but BEFORE splitting.
"""

import re
import pandas as pd
from pathlib import Path


# -------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_clean.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_normalized.parquet"


# -------------------------------------------------------
# QUESTION VALIDATION
# -------------------------------------------------------
def looks_like_question(text: str) -> bool:
    """
    Validate whether a string is a real FAQ question.
    """
    if not isinstance(text, str):
        return False

    stripped = text.strip()

    # Minimum length
    if len(stripped) < 8:
        return False

    # Must end with question mark
    if not stripped.endswith("?"):
        return False

    # Headers frequently mislabeled as questions
    invalid_headers = [
        r"^About\b",
        r"^Overview\b",
        r"^Features\b",
        r"^Eligibility\b",
        r"^You may also like\b",
        r"^Other ways to bank\b",
        r"^Legal\b",
        r"^Privacy\b",
    ]

    for p in invalid_headers:
        if re.match(p, stripped, flags=re.IGNORECASE):
            return False

    return True


# -------------------------------------------------------
# ANSWER NORMALIZATION
# -------------------------------------------------------
def normalize_answer(text: str) -> str:
    """
    Normalize answers without destroying structure.
    Keep paragraphs and line breaks intact.
    """
    if not isinstance(text, str):
        return ""

    # Standardize line breaks
    text = text.replace("\r", "\n")

    # Remove excessive spaces on lines
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]

    # Remove stray markdown bullets at line start
    cleaned_lines = [re.sub(r"^\s*[-•]\s*", "", ln) for ln in lines]

    # Restore as multi-line text
    text = "\n".join(cleaned_lines).strip()

    return text


# -------------------------------------------------------
# ROW NORMALIZATION LOGIC
# -------------------------------------------------------
def normalize_faq_row(question: str, answer: str):
    """
    Normalize a single FAQ row.

    Rules:
        - If question starts like a natural question but lacks '?',
          add the '?'.
        - If question is not valid, discard the row entirely
          (do NOT merge into answer).
        - Clean answer structure.
    """
    q = question.strip() if isinstance(question, str) else ""
    a = answer.strip() if isinstance(answer, str) else ""

    # Add missing question mark if obviously a question
    if q and not q.endswith("?"):
        prefixes = ("how", "what", "why", "when", "where", "who",
                    "can", "does", "is", "are", "do", "should")
        if q.lower().startswith(prefixes):
            q = q.rstrip(".") + "?"

    # If still not a valid question, discard the row
    if not looks_like_question(q):
        return None, None

    # Clean answer while preserving structure
    normalized_a = normalize_answer(a)

    if not normalized_a:
        return None, None

    return q, normalized_a


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def normalize_faqs():
    print(f"Loading {INPUT_PATH.name}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    # Preserve provenance columns
    provenance_cols = [col for col in ["url", "source", "retrieved_at"] if col in df.columns]

    normalized_rows = []

    for _, row in df.iterrows():
        q = row["question"]
        a = row["answer"]

        norm_q, norm_a = normalize_faq_row(q, a)

        # Skip rows with invalid questions
        if not norm_q or not norm_a:
            continue

        entry = {
            "question": norm_q,
            "answer": norm_a,
        }

        # Add provenance fields if present
        for col in provenance_cols:
            entry[col] = row[col]

        normalized_rows.append(entry)

    normalized_df = (
        pd.DataFrame(normalized_rows)
        .drop_duplicates(subset=["question", "answer"])
        .reset_index(drop=True)
    )

    print(f"Normalized to {len(normalized_df)} rows")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved normalized dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    normalize_faqs()
