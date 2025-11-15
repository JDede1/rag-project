"""
validate_scraper.py
-------------------------------------
Official validator for RBC FAQ scraping.

Responsible for verifying:
    • URL coverage (every URL must have at least 1 FAQ)
    • FAQ count distribution (no page should produce <3 FAQs)
    • HTML artifacts in answers
    • Oversized answers (indicates markdown fallback triggered)
    • Boilerplate / low-value content
    • Duplicate answers (bad extraction patterns)
    • Raw JSON integrity

Outputs:
    • Clear validation report
    • Raises RuntimeError if ANY critical checks fail
    • Designed for use in:
          python src/ingestion/validate_scraper.py
"""

import json
import re
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "rbc"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs.parquet"
URL_FILE = Path(__file__).resolve().parent / "rbc_urls.txt"


# ------------------------------------------------------------
# BOILERPLATE DETECTION
# ------------------------------------------------------------
BOILERPLATE_PATTERNS = [
    r"copyright",
    r"cookies",
    r"privacy",
    r"terms",
    r"contact us",
    r"©",
    r"all rights reserved",
    r"follow us",
    r"email",
]


def looks_like_boilerplate(text: str) -> bool:
    if not isinstance(text, str):
        return True
    low_value = any(re.search(p, text.lower()) for p in BOILERPLATE_PATTERNS)
    too_short = len(text.strip()) < 40
    return low_value or too_short


# ------------------------------------------------------------
# VALIDATION PIPELINE
# ------------------------------------------------------------
def validate_scraper():
    print("\n=== VALIDATING RBC SCRAPER OUTPUT ===\n")

    # --------------------------------------------------------
    # 1. Check processed file exists
    # --------------------------------------------------------
    if not PROCESSED_PATH.exists():
        raise RuntimeError("ERROR: Processed file rbc_faqs.parquet not found.")

    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded {len(df)} FAQ rows.")

    errors = []

    # --------------------------------------------------------
    # 2. URL coverage
    # --------------------------------------------------------
    with open(URL_FILE, "r") as f:
        expected_urls = [u.strip() for u in f if u.strip()]

    scraped_urls = df["url"].unique().tolist()

    missing_urls = [u for u in expected_urls if u not in scraped_urls]

    print(f"Expected URL count: {len(expected_urls)}")
    print(f"Scraped URL count:  {len(scraped_urls)}")

    if missing_urls:
        errors.append(f"Missing FAQ entries for {len(missing_urls)} URL(s).")
        print("\nMissing URLs:")
        for u in missing_urls:
            print(f" - {u}")
    else:
        print("All URLs successfully scraped.")


    # --------------------------------------------------------
    # 3. FAQ count distribution (per URL)
    # --------------------------------------------------------
    counts = df.groupby("url").size()
    problematic_counts = counts[counts < 3]

    if len(problematic_counts) > 0:
        errors.append("Some URLs produced very few (<3) FAQ pairs.")
        print("\nURLs with low FAQ output (<3):")
        print(problematic_counts)
    else:
        print("\nAll URLs produced ≥ 3 FAQ entries.")


    # --------------------------------------------------------
    # 4. Duplicate answers (indicates noisy extraction)
    # --------------------------------------------------------
    dup_answers = df["answer"].duplicated(keep=False).sum()
    print(f"\nDuplicate answers detected: {dup_answers}")

    if dup_answers > len(df) * 0.25:
        errors.append("High number of duplicate answers detected.")


    # --------------------------------------------------------
    # 5. Boilerplate / low-value answers
    # --------------------------------------------------------
    boilerplate_count = df["answer"].apply(looks_like_boilerplate).sum()
    print(f"Boilerplate detections: {boilerplate_count}")

    if boilerplate_count / len(df) > 0.10:
        errors.append("More than 10% of answers appear to be boilerplate content.")


    # --------------------------------------------------------
    # 6. Overly long answers (markdown fallback triggered)
    # --------------------------------------------------------
    long_answers = df[df["answer"].str.len() > 1500]
    long_count = len(long_answers)
    print(f"Very long answers (>1500 chars): {long_count}")

    if long_count > 0:
        errors.append(
            "Some answers exceed 1500 characters — markdown fallback likely triggered."
        )


    # --------------------------------------------------------
    # 7. Empty questions or answers
    # --------------------------------------------------------
    empty_questions = df[df["question"].str.strip() == ""]
    empty_answers = df[df["answer"].str.strip() == ""]

    if len(empty_questions) > 0:
        errors.append("Empty questions detected.")

    if len(empty_answers) > 0:
        errors.append("Empty answers detected.")


    # --------------------------------------------------------
    # 8. HTML artifacts inside answers
    # --------------------------------------------------------
    html_artifacts = df["answer"].str.contains(r"<[^>]+>", regex=True).sum()
    print(f"HTML artifact detections: {html_artifacts}")

    if html_artifacts > 0:
        errors.append("Some answers still contain raw HTML tags.")


    # --------------------------------------------------------
    # 9. Validate raw JSON files
    # --------------------------------------------------------
    raw_files = list(RAW_DIR.glob("*.json"))
    invalid_json_count = 0

    for file in raw_files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                invalid_json_count += 1
        except Exception:
            invalid_json_count += 1

    print(f"Corrupted raw JSON files: {invalid_json_count}")

    if invalid_json_count > 0:
        errors.append(f"{invalid_json_count} raw JSON files are corrupted or invalid.")


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------
    if errors:
        print("\n=== SCRAPER VALIDATION FAILED ===")
        for e in errors:
            print(f" - {e}")
        raise RuntimeError("Scraper validation failed. Fix issues before continuing.")
    else:
        print("\n=== SCRAPER VALIDATION PASSED — Dataset is clean and complete ===\n")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    validate_scraper()
