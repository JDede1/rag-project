"""
normalize_faqs.py
-------------------------------------------------------
Normalize FAQ entries so that all question/answer pairs:

    • Follow consistent structure
    • Do not contain navigation artifacts as questions
    • Have correctly aligned "question" and "answer" text
    • Remove section headers that were mistakenly treated as questions
    • Convert multi-line answers into clean paragraphs
    • Prepare the dataset for downstream splitting and chunking

This version preserves provenance fields:
    • url
    • source
    • retrieved_at

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
    Determine whether a string is a valid FAQ-style question.
    """
    if not isinstance(text, str):
        return False

    stripped = text.strip()

    if len(stripped) < 8:
        return False

    if not stripped.endswith("?"):
        return False

    header_patterns = [
        r"^About\b",
        r"^Overview\b",
        r"^Features\b",
        r"^Eligibility\b",
        r"^You may also like\b",
        r"^Other ways to bank\b",
    ]

    for p in header_patterns:
        if re.match(p, stripped, flags=re.IGNORECASE):
            return False

    return True


# -------------------------------------------------------
# ANSWER NORMALIZATION
# -------------------------------------------------------
def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Remove markdown bullets converted to text by scraper
    text = re.sub(r"^\s*[-•]\s*", "", text)

    return text.strip()


# -------------------------------------------------------
# STRUCTURAL NORMALIZATION
# -------------------------------------------------------
def normalize_faq_row(question: str, answer: str):
    """
    Normalize a single FAQ row, fixing common scraping issues.
    """
    q = question.strip() if isinstance(question, str) else ""
    a = answer.strip() if isinstance(answer, str) else ""

    # Case 1: Add missing question mark if it's obviously a question
    if not q.endswith("?"):
        if q.lower().startswith(("how", "what", "why", "when", "where", "who", "can", "does", "is", "are", "do")):
            q = q.rstrip(".") + "?"

    # Case 2: If the question is not real, demote it to answer
    if not looks_like_question(q):
        merged_answer = (q + " " + a).strip()
        return "", normalize_answer(merged_answer)

    # Case 3: Normalize answer
    normalized_a = normalize_answer(a)

    return q, normalized_a


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def normalize_faqs():
    print(f"Loading {INPUT_PATH.name}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    # Identify provenance columns from the cleaned dataset
    provenance_cols = []
    for col in ["url", "source", "retrieved_at"]:
        if col in df.columns:
            provenance_cols.append(col)

    normalized_rows = []

    for _, row in df.iterrows():
        q = row["question"]
        a = row["answer"]

        norm_q, norm_a = normalize_faq_row(q, a)

        if norm_q and norm_a:
            entry = {
                "question": norm_q,
                "answer": norm_a,
            }

            # Add provenance values if present
            for col in provenance_cols:
                entry[col] = row[col]

            normalized_rows.append(entry)

    normalized_df = pd.DataFrame(normalized_rows).drop_duplicates(subset=["question", "answer"])

    print(f"Normalized to {len(normalized_df)} rows")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved normalized dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    normalize_faqs()
