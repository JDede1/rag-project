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

This module should run AFTER cleaning but BEFORE splitting.
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

    Rules:
        • Must end with a question mark
        • Must contain at least a small amount of text
        • Should not be section headers or navigation items
    """

    if not isinstance(text, str):
        return False

    stripped = text.strip()

    if len(stripped) < 8:
        return False

    if not stripped.endswith("?"):
        return False

    # Exclude common RBC section headers that the scraper may extract
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
    """
    Normalize answer text:
        • Collapse multiple spaces
        • Remove stray markdown artifacts
        • Convert multi-line fragments into paragraphs
    """

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
    Normalize a single FAQ row, attempting to fix common scraping issues.

    Cases handled:
        • Missing question marks corrected where appropriate
        • Section headers mistakenly placed in "question" are moved into answer
        • Answers beginning with a question-like fragment are corrected
    """

    q = question.strip() if isinstance(question, str) else ""
    a = answer.strip() if isinstance(answer, str) else ""

    # Case 1: Question does not end with "?" but appears to be a question.
    if not q.endswith("?"):
        if q.lower().startswith(("how", "what", "why", "when", "where", "who", "can", "does", "is", "are", "do")):
            q = q.rstrip(".") + "?"

    # Case 2: If the question is not a real question, demote it.
    if not looks_like_question(q):
        # Treat it as a heading and append to the answer
        merged_answer = (q + " " + a).strip()
        # Provide a placeholder question only as fallback
        # (this will likely be caught later in validation)
        return "", normalize_answer(merged_answer)

    # Case 3: Normalize the answer
    normalized_a = normalize_answer(a)

    return q, normalized_a


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------

def normalize_faqs():
    print(f"Loading {INPUT_PATH.name}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    normalized_rows = []

    for _, row in df.iterrows():
        q, a = normalize_faq_row(row["question"], row["answer"])
        if q.strip() and a.strip():
            normalized_rows.append({"question": q.strip(), "answer": a.strip()})

    normalized_df = pd.DataFrame(normalized_rows).drop_duplicates(subset=["question", "answer"])

    print(f"Normalized to {len(normalized_df)} rows")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved normalized dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    normalize_faqs()
