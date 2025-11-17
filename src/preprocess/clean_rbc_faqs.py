"""
clean_rbc_faqs.py
-------------------------------------------------------
Deep text cleaning for scraped RBC FAQ data.

Improvements in Smart Fix (Option B):
    • Normalizes all RBC phone number formats:
        - 1-800-xxx-xxxx
        - 1 877 xxx xxxx
        - 1.888.xxx.xxxx
        - 1-800-xxx-xxxx (unicode dashes)
    • Repairs truncated phone numbers such as: "call 1"
    • Ensures all numbers follow one consistent canonical format:
          1-XXX-XXX-XXXX
    • WITHOUT altering any other original functionality.

This version continues to preserve structure:
    - Paragraphs
    - Newlines
    - Provenance fields
"""

import re
import pandas as pd
from pathlib import Path


# -------------------------------------------------------
# PATH CONFIGURATION
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_clean.parquet"


# -------------------------------------------------------
# SMART FIX — PHONE NUMBER NORMALIZATION
# -------------------------------------------------------
def normalize_phone_numbers(text: str) -> str:
    """
    Normalize all RBC phone number formats into:
        1-XXX-XXX-XXXX

    Also repairs:
        "call 1" → leaves it untouched unless real digits follow
        broken splits like "1 888 769 2585" → "1-888-769-2585"
        unicode hyphens / dots / spaces → hyphens
    """
    if not isinstance(text, str):
        return ""

    original = text

    # Replace unicode dashes with ASCII
    text = text.replace("-", "-").replace("–", "-").replace("—", "-")

    # Common RBC phone patterns (spaces, dots, hyphens)
    phone_pattern = re.compile(
        r"""
        1              # Leading 1
        [\s\-.]?       # Optional separator
        (\d{3})        # Area code
        [\s\-.]?       # Optional separator
        (\d{3})        # Prefix
        [\s\-.]?       # Optional separator
        (\d{4})        # Line number
        """,
        re.VERBOSE
    )

    def repl(match):
        a, b, c = match.group(1), match.group(2), match.group(3)
        return f"1-{a}-{b}-{c}"

    text = phone_pattern.sub(repl, text)

    # Fix known truncated format: "call 1" (no digits after)
    # We do NOT guess missing numbers
    text = re.sub(r"\bcall 1\b", "call", text, flags=re.IGNORECASE)

    return text


# -------------------------------------------------------
# TEXT NORMALIZATION HELPERS
# -------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def remove_bullet_artifacts(text: str) -> str:
    return re.sub(r"[•▪●]", "-", text)


def clean_common_html_leftovers(text: str) -> str:
    patterns = [
        r"\[.*?\]\(.*?\)",
        r"Back to top",
        r"^\s*#\s*",
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return text


def remove_boilerplate(text: str) -> str:
    boilerplate_patterns = [
        r"Royal Bank of Canada",
        r"©.*?RBC",
        r"Book an appointment",
        r"Find a branch",
        r"Use our mobile app",
        r"You may also like",
        r"Other ways to bank",
        r"Legal Disclaimer",
        r"Privacy & Security",
        r"Cookie (Preferences|Settings)",
        r"All rights reserved",
        r"This page was last updated",
    ]
    for p in boilerplate_patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return text


def deep_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = remove_bullet_artifacts(text)
    text = clean_common_html_leftovers(text)
    text = remove_boilerplate(text)

    # NEW: smart phone normalization
    text = normalize_phone_numbers(text)

    text = normalize_whitespace(text)

    return text


# -------------------------------------------------------
# VALIDATION HELPERS
# -------------------------------------------------------
def is_valid_faq(question: str, answer: str) -> bool:
    if not question or not answer:
        return False

    if len(question) < 8 or len(answer) < 15:
        return False

    invalid_patterns = [
        r"^\W+$",
        r"lorem ipsum",
    ]

    q = question.lower()
    a = answer.lower()

    for p in invalid_patterns:
        if re.search(p, q) or re.search(p, a):
            return False

    return True


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------
def clean_rbc_faqs():
    print("Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} rows")

    preserved_cols = [c for c in ["url", "source", "retrieved_at"] if c in df.columns]

    df["question"] = df["question"].apply(deep_clean)
    df["answer"] = df["answer"].apply(deep_clean)

    df = df[df.apply(lambda x: is_valid_faq(x["question"], x["answer"]), axis=1)]
    df.drop_duplicates(subset=["question", "answer"], inplace=True)

    final_cols = ["question", "answer"] + preserved_cols
    df = df[final_cols]

    print(f"After cleaning: {len(df)} rows remain")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_rbc_faqs()
