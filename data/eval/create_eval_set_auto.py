"""
create_eval_set_auto.py
==================================

Automatically generates the evaluation set from the REAL
RBC processed dataset created in Phases 1–2.

Output:
    data/eval/rbc_eval_set.jsonl

Contents:
    • 20 automatically selected REAL RBC known Q/A
    • 10 unknown questions (bank-irrelevant)
"""

import json
from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
REFINED_PARQUET = Path("data/processed/rbc_faqs_refined.parquet")
OUTPUT_FILE = Path("data/eval/rbc_eval_set.jsonl")

# Number of known (real RBC) questions to extract
KNOWN_LIMIT = 20

# Predefined unknown questions (bank-irrelevant)
UNKNOWN_QUESTIONS = [
    ("u01", "What is the weather in Toronto today?"),
    ("u02", "How do I open a chequing account with TD Bank?"),
    ("u03", "Does CIBC offer cryptocurrency trading?"),
    ("u04", "What are the Scotiabank student account fees?"),
    ("u05", "How do I get a refund for a Steam purchase?"),
    ("u06", "Can I use my RBC debit card on Mars?"),
    ("u07", "How can I close my Chase Sapphire credit card?"),
    ("u08", "What is the interest rate on a Bank of America Rewards card?"),
    ("u09", "What are the hours for the branch inside Walmart?"),
    ("u10", "How do I apply for a Wells Fargo auto loan?"),
]


# ------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------
def load_refined_dataset():
    """Load Phase 2 final refined dataset."""
    if not REFINED_PARQUET.exists():
        raise FileNotFoundError(
            f"ERROR: Missing {REFINED_PARQUET}\n"
            "Run Phases 1–2 first to generate it."
        )
    return pd.read_parquet(REFINED_PARQUET)


def extract_unique_questions(df):
    """Returns unique (question, answer) pairs from refined RBC dataset."""
    df = df[["question", "answer"]].dropna().copy()

    # Combine question + answer to dedupe uniquely
    df["qa_key"] = df["question"].str.strip() + "||" + df["answer"].str.strip()
    df = df.drop_duplicates(subset=["qa_key"])

    return df[["question", "answer"]]


def select_known_examples(df, limit):
    """Take first N unique RBC questions as evaluation set."""
    df = df.head(limit).copy()

    eval_records = []
    for i, row in enumerate(df.itertuples(), start=1):
        eval_records.append({
            "id": f"k{i:02d}",
            "type": "known",
            "question": row.question,
            "answer": row.answer
        })

    return eval_records


def build_unknown_examples():
    """Generate unknown questions with null answers."""
    unknown = []
    for qid, q in UNKNOWN_QUESTIONS:
        unknown.append({
            "id": qid,
            "type": "unknown",
            "question": q,
            "answer": None
        })
    return unknown


def write_jsonl(records):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Created evaluation set: {OUTPUT_FILE}")
    print(f"Total examples: {len(records)}\n")


def main():
    print("\n=== AUTO GENERATE RBC EVAL SET ===")

    df = load_refined_dataset()
    unique_qa = extract_unique_questions(df)

    if unique_qa.empty:
        raise ValueError("No Q/A pairs found in refined dataset.")

    print(f"Unique RBC Q/A pairs available: {len(unique_qa)}")

    known = select_known_examples(unique_qa, KNOWN_LIMIT)
    unknown = build_unknown_examples()

    all_records = known + unknown

    write_jsonl(all_records)

    print("Sample known record:")
    print(json.dumps(all_records[0], indent=2))


if __name__ == "__main__":
    main()
