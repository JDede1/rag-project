"""
diagnose_scraper.py
-------------------------------------
Deep diagnostic tool for RBC scraper output.

This script helps identify:
    • Which URLs triggered markdown fallback extraction
    • Which URLs produced <3 FAQ entries
    • Which URLs produced extremely long answers (>1500)
    • Duplicate questions per URL
    • Sample Q/A per URL
    • Missing URLs
    • Per-URL FAQ counts

Run:
    python src/ingestion/diagnose_scraper.py
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "rbc"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs.parquet"
URL_FILE = Path(__file__).resolve().parent / "rbc_urls.txt"


# ------------------------------------------------------------
# MAIN DIAGNOSTIC
# ------------------------------------------------------------
def diagnose_scraper():
    print("\n=== RBC SCRAPER DIAGNOSTICS ===\n")

    # --------------------------------------------------------
    # Load canonical URL list
    # --------------------------------------------------------
    with open(URL_FILE, "r") as f:
        expected_urls = [u.strip() for u in f if u.strip()]

    print(f"Loaded {len(expected_urls)} expected URLs.\n")

    # --------------------------------------------------------
    # Load processed parquet
    # --------------------------------------------------------
    if not PROCESSED_PATH.exists():
        raise RuntimeError("Processed file rbc_faqs.parquet does not exist.")

    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded {len(df)} extracted FAQ rows.\n")

    # URL → list of FAQ entries
    url_groups = dict(tuple(df.groupby("url")))

    # --------------------------------------------------------
    # Determine missing URLs
    # --------------------------------------------------------
    missing_urls = [u for u in expected_urls if u not in url_groups]
    if missing_urls:
        print("❌ Missing URLs (no FAQs extracted):")
        for u in missing_urls:
            print(f" - {u}")
    else:
        print("✔ All URLs produced at least one FAQ.\n")

    # --------------------------------------------------------
    # Per-URL FAQ counts
    # --------------------------------------------------------
    print("\n=== FAQ Count Per URL ===")
    for url in expected_urls:
        count = len(url_groups.get(url, []))
        flag = " (LOW)" if count < 3 else ""
        print(f"{count:3d}  -  {url}{flag}")

    # --------------------------------------------------------
    # Detect fallback: answers starting with fallback label
    # --------------------------------------------------------
    print("\n=== URLs Triggering Fallback (Markdown Extraction) ===")

    fallback_hits = defaultdict(int)
    for _, row in df.iterrows():
        if row["question"].startswith("Full Page Content"):
            fallback_hits[row["url"]] += 1

    if fallback_hits:
        for url, count in fallback_hits.items():
            print(f"❌ {url} → {count} fallback entries")
    else:
        print("✔ No fallback entries detected.")

    # --------------------------------------------------------
    # Detect extremely long answers (>1500 chars)
    # --------------------------------------------------------
    print("\n=== URLs with Long Answers (>1500 chars) ===")

    long_answer_urls = defaultdict(int)
    for _, row in df.iterrows():
        if len(row["answer"]) > 1500:
            long_answer_urls[row["url"]] += 1

    if long_answer_urls:
        for url, count in long_answer_urls.items():
            print(f"❌ {url} → {count} long answers")
    else:
        print("✔ No excessively long answers detected.")

    # --------------------------------------------------------
    # Detect duplicate questions per URL
    # --------------------------------------------------------
    print("\n=== Duplicate Questions per URL ===")

    dup_urls = []
    for url, subdf in url_groups.items():
        dup_count = subdf["question"].duplicated(keep=False).sum()
        if dup_count > 0:
            dup_urls.append((url, dup_count))

    if dup_urls:
        for url, count in dup_urls:
            print(f"❌ {url} → {count} duplicate questions")
    else:
        print("✔ No duplicate questions detected.")

    # --------------------------------------------------------
    # Sample Q/A per URL
    # --------------------------------------------------------
    print("\n=== Sample Q/A per URL ===")

    for url in expected_urls:
        subset = url_groups.get(url)
        if subset is None or len(subset) == 0:
            print(f"\n--- {url} (NO DATA) ---")
            continue

        print(f"\n--- {url} ({len(subset)} FAQ entries) ---")

        sample = subset.head(2)

        for i, row in sample.iterrows():
            print(f"\nQ: {row['question'][:200]}")
            print(f"A: {row['answer'][:300]}\n")

    print("\n=== DIAGNOSTICS COMPLETE ===\n")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    diagnose_scraper()
