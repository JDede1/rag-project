"""
clean_rbc_faqs.py
-------------------------------------------------------
Deep text cleaning for scraped RBC FAQ data.

This version preserves structural formatting:
    - Keeps paragraph boundaries
    - Keeps newline separation
    - Normalizes spaces without flattening structure

Also preserves provenance fields:
    - url
    - source
    - retrieved_at
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
# TEXT NORMALIZATION HELPERS
# -------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    """
    Normalize excessive spaces while preserving line breaks.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\xa0", " ")     # non-breaking spaces
    text = text.replace("\u200b", "")    # zero-width spaces

    # Normalize repeated spaces but DO NOT collapse newlines
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3+ consecutive newlines with exactly 2 (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip trailing spaces on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def remove_bullet_artifacts(text: str) -> str:
    """
    Replace unusual bullets with standard dash but preserve line structure.
    """
    return re.sub(r"[•▪●]", "-", text)


def clean_common_html_leftovers(text: str) -> str:
    """
    Remove common scraper artifacts without damaging content.
    """
    patterns = [
        r"\[.*?\]\(.*?\)",     # markdown links
        r"Back to top",
        r"^\s*#\s*",           # markdown headers
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return text


def remove_boilerplate(text: str) -> str:
    """
    Remove navigation/footer boilerplate that should never appear in answers.
    """
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
    """
    Apply layered cleaning without destroying structure.
    """
    if not isinstance(text, str):
        return ""

    text = remove_bullet_artifacts(text)
    text = clean_common_html_leftovers(text)
    text = remove_boilerplate(text)
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

    # Preserve provenance columns if present
    preserved_cols = [col for col in ["url", "source", "retrieved_at"] if col in df.columns]

    # Clean Q/A fields independently
    df["question"] = df["question"].apply(deep_clean)
    df["answer"] = df["answer"].apply(deep_clean)

    # Drop invalid rows
    df = df[df.apply(lambda x: is_valid_faq(x["question"], x["answer"]), axis=1)]

    # Remove exact duplicates
    df.drop_duplicates(subset=["question", "answer"], inplace=True)

    # Reorder to final schema
    final_cols = ["question", "answer"] + preserved_cols
    df = df[final_cols]

    print(f"After cleaning: {len(df)} rows remain")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_rbc_faqs()
