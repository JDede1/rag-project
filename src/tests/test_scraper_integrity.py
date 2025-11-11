"""
test_scraper_integrity.py
-------------------------------------
Quick integrity and quality check for scraped FAQ datasets.

Purpose:
    • Verify that the scraped dataset exists and loads successfully
    • Check for duplicates, missing values, and extreme text lengths
    • Compute basic statistics (mean, min, max lengths)
    • Print random sample FAQs for visual verification

Usage:
    python src/tests/test_scraper_integrity.py data/processed/rbc_faqs.parquet
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime


# ----------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------
def describe_text_lengths(df, col):
    """Return summary statistics for text column lengths."""
    lengths = df[col].str.len()
    return {
        "min": lengths.min(),
        "max": lengths.max(),
        "mean": round(lengths.mean(), 2),
        "median": lengths.median(),
    }


def check_dataset(file_path):
    """Perform dataset integrity checks."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    df = pd.read_parquet(path)
    print(f"\n📦 Loaded dataset: {path.name}")
    print(f"🕒 Checked on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # --- Basic Checks ---
    print("🧩 Basic Integrity Checks:")
    print(f"• Missing values: {df.isnull().sum().sum()}")
    print(f"• Duplicates: {df.duplicated(subset=['question', 'answer']).sum()}")
    print(f"• Unique URLs: {df['url'].nunique() if 'url' in df else 'N/A'}")
    print()

    # --- Text Length Stats ---
    print("✏️ Text Length Statistics:")
    for col in ["question", "answer"]:
        stats = describe_text_lengths(df, col)
        print(f"• {col.capitalize()}: min={stats['min']}, max={stats['max']}, mean={stats['mean']}, median={stats['median']}")
    print()

    # --- Sanity Rules ---
    too_short = df[(df["question"].str.len() < 10) | (df["answer"].str.len() < 30)]
    too_long = df[(df["answer"].str.len() > 2000)]
    print(f"⚠️ Too-short entries: {len(too_short)}")
    print(f"⚠️ Too-long entries: {len(too_long)}\n")

    # --- Sample Preview ---
    print("🔍 Sample FAQs:")
    sample_df = df.sample(n=min(5, len(df)), random_state=42)
    for i, row in sample_df.iterrows():
        print(f"Q: {row['question']}\nA: {row['answer'][:200]}...\n{'-'*50}")

    # --- Verdict ---
    print("\n✅ Integrity check completed.")
    if len(df) < 5:
        print("⚠️ WARNING: Dataset too small. Check your scraper or URLs.")
    if len(too_short) > 0 or len(too_long) > 0:
        print("⚠️ Some entries may need cleaning or further filtering.")


# ----------------------------------------
# MAIN ENTRY
# ----------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/tests/test_scraper_integrity.py <path_to_parquet>")
        sys.exit(1)

    check_dataset(sys.argv[1])
