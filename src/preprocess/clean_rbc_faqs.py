"""
clean_rbc_faqs.py
-------------------------------------
Cleans and filters scraped RBC FAQ data for downstream embedding.
"""

import re
import pandas as pd
from pathlib import Path

# -------------------------
# PATHS
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_clean.parquet"


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def clean_text(text: str) -> str:
    """Remove HTML artifacts, excessive spaces, and non-printable characters."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = text.strip()
    return text


def is_valid_entry(q, a) -> bool:
    """Basic validation rules for good FAQs."""
    if len(q) < 10 or len(a) < 20:
        return False
    # filter out noise patterns
    noise_patterns = [
        "cookie", "privacy", "footer", "email", "error",
        "sign up", "notification", "link", "manage your cookie",
        "rbc.com", "©"
    ]
    joined = (q + " " + a).lower()
    if any(word in joined for word in noise_patterns):
        return False
    return True


# -------------------------
# MAIN CLEANING PIPELINE
# -------------------------
def clean_rbc_faqs():
    print("🔹 Loading dataset...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} records")

    # Clean text columns
    df["question"] = df["question"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text)

    # Drop invalid rows
    df = df[df.apply(lambda x: is_valid_entry(x["question"], x["answer"]), axis=1)]

    # Remove duplicates
    df.drop_duplicates(subset=["question", "answer"], inplace=True)

    print(f"✅ After cleaning: {len(df)} records remain")

    # Save cleaned dataset
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"💾 Saved cleaned data → {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_rbc_faqs()
