"""
clean_rbc_faqs.py
-------------------------------------------------------
Deep text cleaning for scraped RBC FAQ data.

Goals:
    • Remove boilerplate (navigation, footer, legal text)
    • Normalize whitespace and punctuation
    • Preserve all valid financial Q/A content
    • Avoid over-filtering legitimate questions
    • Output clean Q/A pairs for downstream normalization
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
    if not isinstance(text, str):
        return ""

    text = text.replace("\xa0", " ")     # non-breaking spaces
    text = text.replace("\u200b", "")    # zero-width spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_bullet_artifacts(text: str) -> str:
    text = re.sub(r"[•▪●]", "-", text)   # standardize bullets
    return text


def clean_common_html_leftovers(text: str) -> str:
    # Remove common markdown/html conversion artifacts
    patterns = [
        r"http[s]?:\/\/\S+",      # URLs
        r"\[.*?\]\(.*?\)",        # markdown links
        r"Back to top",           # navigation fragments
        r"^\s*#\s*",              # markdown headers (#)
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return text


def remove_boilerplate(text: str) -> str:
    """
    RBC site boilerplate patterns that frequently contaminate scraped content.
    These were identified empirically from RBC FAQ pages.
    """
    boilerplate_patterns = [
        r"Royal Bank of Canada", 
        r"©.*?RBC", 
        r"Use our mobile app", 
        r"Book an appointment",
        r"Find a branch",
        r"Contact us",
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

    text = normalize_whitespace(text)
    text = remove_bullet_artifacts(text)
    text = clean_common_html_leftovers(text)
    text = remove_boilerplate(text)

    # Final pass: collapse spacing again
    text = normalize_whitespace(text)
    return text


# -------------------------------------------------------
# VALIDATION HELPERS
# -------------------------------------------------------
def is_valid_faq(question: str, answer: str) -> bool:
    if not question or not answer:
        return False

    # Minimum lengths
    if len(question) < 8 or len(answer) < 15:
        return False

    # Avoid placeholder or malformed entries
    invalid_patterns = [
        r"^\W+$",      # punctuation-only
        r"lorem ipsum"
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

    # Step 1: deep clean
    df["question"] = df["question"].apply(deep_clean)
    df["answer"] = df["answer"].apply(deep_clean)

    # Step 2: remove invalid rows
    df = df[df.apply(lambda x: is_valid_faq(x["question"], x["answer"]), axis=1)]

    # Step 3: remove duplicates
    df.drop_duplicates(subset=["question", "answer"], inplace=True)

    print(f"After cleaning: {len(df)} rows remain")

    # Step 4: persist
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_rbc_faqs()
