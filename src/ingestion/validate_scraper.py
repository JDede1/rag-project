"""
validate_scraper.py
-------------------------------------
JSON-aware validator for RBC FAQ scraping.

Validates:
    • URL coverage (all URLs scraped)
    • FAQ count distribution (≥3 per URL, unless fallback)
    • Duplicate questions & answers
    • Boilerplate / low-value answers
    • Oversized answers (markdown fallback detection)
    • HTML artifacts in answers
    • Empty Q/A fields
    • Raw JSON integrity

Compatible with:
    • JSON-driven scrape_rbc_faqs.py
    • faq_extraction_patterns.json
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
PATTERN_FILE = Path(__file__).resolve().parent / "faq_extraction_patterns.json"


# ------------------------------------------------------------
# LOAD JSON RULES
# ------------------------------------------------------------
with open(PATTERN_FILE, "r", encoding="utf-8") as f:
    patterns = json.load(f)

rules = patterns["extraction"]


# ------------------------------------------------------------
# BOILERPLATE DETECTION
# ------------------------------------------------------------
def looks_like_boilerplate(text: str) -> bool:
    """
    Identify low-value answers based on:
        • Boilerplate patterns
        • Excessively short content
    """
    if not isinstance(text, str):
        return True

    too_short = len(text.strip()) < 40
    low_value = any(
        re.search(p, text.lower()) for p in [
            "cookie",
            "privacy",
            "terms",
            "footer",
            "disclaimer",
            "all rights reserved",
            "©",
            "javascript",
            "contact us",
            "follow us"
        ]
    )

    return too_short or low_value


# ------------------------------------------------------------
# VALIDATION PIPELINE
# ------------------------------------------------------------
def validate_scraper():
    print("\n=== VALIDATING RBC SCRAPER OUTPUT ===\n")

    errors = []

    # --------------------------------------------------------
    # 1. Check processed file exists
    # --------------------------------------------------------
    if not PROCESSED_PATH.exists():
        raise RuntimeError("ERROR: Could not find data/processed/rbc_faqs.parquet")

    df = pd.read_parquet(PROCESSED_PATH)
    print(f"Loaded {len(df)} FAQ rows.\n")

    # --------------------------------------------------------
    # 2. URL coverage
    # --------------------------------------------------------
    with open(URL_FILE, "r") as f:
        expected_urls = [u.strip() for u in f if u.strip()]

    scraped_urls = sorted(df["url"].unique().tolist())

    missing_urls = [u for u in expected_urls if u not in scraped_urls]

    print(f"Expected URL count: {len(expected_urls)}")
    print(f"Scraped URL count:  {len(scraped_urls)}")

    if missing_urls:
        errors.append(f"Missing FAQ entries for {len(missing_urls)} URL(s).")
        print("\nMissing URLs:")
        for u in missing_urls:
            print(" -", u)
    else:
        print("All URLs successfully scraped.")

    # --------------------------------------------------------
    # 3. FAQ count distribution
    # --------------------------------------------------------
    counts = df.groupby("url").size()

    low_output = counts[counts < 3]
    if len(low_output) > 0:
        errors.append("Some URLs produced very few (<3) FAQ entries.")
        print("\nURLs with low FAQ counts (<3):")
        print(low_output)
    else:
        print("\nAll URLs produced ≥ 3 FAQ entries.")

    # --------------------------------------------------------
    # 4. Duplicate questions/answers
    # --------------------------------------------------------
    dup_questions = df["question"].duplicated(keep=False).sum()
    dup_answers = df["answer"].duplicated(keep=False).sum()

    print(f"\nDuplicate questions detected: {dup_questions}")
    print(f"Duplicate answers detected: {dup_answers}")

    if dup_answers > len(df) * 0.25:
        errors.append("High number of duplicate answers detected (possible extraction noise).")

    # --------------------------------------------------------
    # 5. Boilerplate content
    # --------------------------------------------------------
    boilerplate_count = df["answer"].apply(looks_like_boilerplate).sum()
    print(f"\nBoilerplate detections: {boilerplate_count}")

    if boilerplate_count / len(df) > 0.10:
        errors.append("More than 10% of answers appear to be boilerplate.")

    # --------------------------------------------------------
    # 6. Very long answers (markdown fallback triggered)
    # --------------------------------------------------------
    max_len = rules["max_fallback_answer_chars"]
    long_answers = df[df["answer"].str.len() > max_len]
    long_count = len(long_answers)

    print(f"Very long answers (> {max_len} chars): {long_count}")

    if long_count > 0:
        errors.append(
            f"{long_count} answers exceed fallback limit ({max_len} chars) — markdown fallback likely triggered."
        )

    # --------------------------------------------------------
    # 7. Empty Q/A fields
    # --------------------------------------------------------
    empty_q = df[df["question"].str.strip() == ""]
    empty_a = df[df["answer"].str.strip() == ""]

    if len(empty_q) > 0:
        errors.append("Empty questions detected.")

    if len(empty_a) > 0:
        errors.append("Empty answers detected.")

    # --------------------------------------------------------
    # 8. HTML artifacts
    # --------------------------------------------------------
    html_failures = df["answer"].str.contains(r"<[^>]+>", regex=True).sum()
    print(f"HTML artifact detections: {html_failures}")

    if html_failures > 0:
        errors.append("Some answers contain raw HTML tags.")

    # --------------------------------------------------------
    # 9. Raw JSON integrity
    # --------------------------------------------------------
    raw_files = list(RAW_DIR.glob("*.json"))
    invalid_json = 0

    for file in raw_files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                invalid_json += 1
        except Exception:
            invalid_json += 1

    print(f"Corrupted raw JSON files: {invalid_json}")

    if invalid_json > 0:
        errors.append(f"{invalid_json} raw JSON files are corrupted or invalid.")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------
    print("\n-------------------------------------")

    if errors:
        print("=== SCRAPER VALIDATION FAILED ===")
        for e in errors:
            print(" -", e)
        print("-------------------------------------\n")
        raise RuntimeError("Validation failed — scraper output is not clean.")
    else:
        print("=== SCRAPER VALIDATION PASSED — Dataset is clean and complete ===")
        print("-------------------------------------\n")


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    validate_scraper()
