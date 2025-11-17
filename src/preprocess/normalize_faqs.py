"""
normalize_faqs.py
-------------------------------------------------------
Normalize FAQ entries so that all question/answer pairs:

    - Follow consistent question structure
    - Remove section headers / navigation artifacts
    - Repair missing question marks for interrogative sentences
    - Preserve provenance metadata
    - Prepare the dataset for splitting and chunking

Runs AFTER clean_rbc_faqs.py and BEFORE split_compound_faqs.py.
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
# HEADER / NAVIGATION DETECTION
# -------------------------------------------------------
HEADER_PATTERNS = [
    r"^about\b",
    r"^overview\b",
    r"^features\b",
    r"^eligibility\b",
    r"^general questions\b",
    r"^general questions & concerns\b",
    r"^support\b",
    r"^help\b",
    r"^you may also like\b",
    r"^other ways to bank\b",
    r"^legal\b",
    r"^privacy\b",
    r"^contact\b",
]


def is_section_header(text: str) -> bool:
    """
    Detect headings or navigation text that should not be treated as questions.
    """
    if not isinstance(text, str):
        return False

    t = text.strip().lower()
    for p in HEADER_PATTERNS:
        if re.match(p, t):
            return True

    # Short or generic non-questions
    if len(t) < 8:
        return True

    # Generic labels (common RBC)
    if t in ["faqs", "faq", "questions", "general", "information"]:
        return True

    return False


# -------------------------------------------------------
# QUESTION VALIDATION (SMART MODE)
# -------------------------------------------------------
QUESTION_PREFIXES = (
    "how", "what", "why", "when", "where", "who",
    "can", "could", "does", "do", "is", "are",
    "should", "will", "would", "may", "might"
)


def looks_like_real_question(text: str) -> bool:
    """
    Smart-mode question validator:
      - Must not be a header
      - Must be at least moderate length
      - Must be interrogative or end with '?'
    """
    if not isinstance(text, str):
        return False

    t = text.strip()
    t_lower = t.lower()

    # Reject section headers before anything else
    if is_section_header(t_lower):
        return False

    # End with question mark: almost always valid
    if t.endswith("?") and len(t) >= 6:
        return True

    # No question mark → check interrogative pattern
    if any(t_lower.startswith(pref) for pref in QUESTION_PREFIXES):
        return True

    return False


# -------------------------------------------------------
# ANSWER NORMALIZATION
# -------------------------------------------------------
def normalize_answer(text: str) -> str:
    """
    Normalize answers while preserving paragraph boundaries.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        line = re.sub(r"[ \t]+", " ", line)
        line = re.sub(r"^\s*[-•]\s*", "", line)
        if line.strip():
            lines.append(line.strip())

    return "\n".join(lines).strip()


# -------------------------------------------------------
# SINGLE ROW NORMALIZATION LOGIC
# -------------------------------------------------------
def normalize_faq_row(question: str, answer: str):
    """
    Smart-mode row normalizer:
      - Skip section headers (never merge into answers)
      - Repair missing '?' for interrogative sentences
      - Ensure the answer is non-empty
    """
    q = question.strip() if isinstance(question, str) else ""
    a = answer.strip() if isinstance(answer, str) else ""

    if not q or not a:
        return None, None

    q_lower = q.lower().strip()

    # Reject section headers immediately
    if is_section_header(q_lower):
        return None, None

    # Add missing '?' when it's clearly a question
    if not q.endswith("?"):
        if any(q_lower.startswith(pref) for pref in QUESTION_PREFIXES):
            q = q.rstrip(".") + "?"

    # Final real-question validation
    if not looks_like_real_question(q):
        return None, None

    # Normalize answer
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

    provenance_cols = [c for c in ["url", "source", "retrieved_at"] if c in df.columns]

    normalized_rows = []

    for _, row in df.iterrows():
        q = row["question"]
        a = row["answer"]

        norm_q, norm_a = normalize_faq_row(q, a)

        if not norm_q or not norm_a:
            continue

        entry = {
            "question": norm_q,
            "answer": norm_a,
        }

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
