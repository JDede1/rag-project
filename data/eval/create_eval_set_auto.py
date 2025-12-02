"""
create_eval_set_auto.py
-----------------------------------------------------------
Automatically generates an RBC evaluation set using the
refined dataset produced in Phase 2.

Output:
    data/eval/rbc_eval_set.jsonl

Includes:
    • Known questions (sampled from actual RBC FAQs)
    • Unknown questions (foreign-bank / unrelated queries)
"""

import json
import random
from pathlib import Path
import pandas as pd


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
REFINED_FILE = Path("data/processed/rbc_faqs_refined.parquet")
OUTPUT_FILE = Path("data/eval/rbc_eval_set.jsonl")

N_KNOWN = 60       # number of known questions to sample
N_UNKNOWN = 20     # number of unknown / out-of-domain questions


# -----------------------------------------------------------
# UNKNOWN QUESTIONS (fixed list)
# -----------------------------------------------------------
UNKNOWN_QUESTIONS = [
    "How do I open a chequing account with TD Bank?",
    "What are the Scotiabank student account fees?",
    "Does CIBC offer cryptocurrency trading?",
    "What is the interest rate on a Bank of America Rewards card?",
    "How can I close my Chase Sapphire credit card?",
    "How do I apply for a Wells Fargo auto loan?",
    "What are the hours for the branch inside Walmart?",
    "What is the weather in Toronto today?",
    "Can I use my debit card on Mars?",
    "How do I get a refund for a Steam purchase?",
    "What are the RBC rewards for travel points in 2035?",
    "How do I sign up for Amazon Prime?",
    "How do I fix my Windows 11 PC?",
    "What time does McDonald's close?",
    "How do I update my Netflix billing info?",
    "What is the price of Bitcoin today?",
    "Does NASA offer personal bank accounts?",
    "Where can I buy used furniture in Ottawa?",
    "How do I become a pilot?",
    "Can I withdraw money from an ATM in the moon colony?",
]


# -----------------------------------------------------------
# MAIN LOGIC
# -----------------------------------------------------------
def main():
    print("\n=== AUTO-GENERATED RBC EVAL SET CREATOR ===\n")

    # -------------------------------------------------------
    # 1. LOAD REFINED DATASET
    # -------------------------------------------------------
    if not REFINED_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {REFINED_FILE}\n"
            "Run Phase 2 preprocessing before generating eval set."
        )

    df = pd.read_parquet(REFINED_FILE)

    if not {"question", "answer"}.issubset(df.columns):
        raise ValueError(
            "Refined dataset must contain 'question' and 'answer' columns."
        )

    print(f"Loaded refined dataset: {len(df)} Q/A pairs")

    # -------------------------------------------------------
    # 2. SAMPLE KNOWN QUESTIONS
    # -------------------------------------------------------
    df_sampled = df.sample(n=min(N_KNOWN, len(df)), random_state=42)

    known_items = []
    for i, row in df_sampled.iterrows():
        known_items.append({
            "id": f"k{i}",
            "type": "known",
            "question": row["question"],
            "answer": row["answer"]
        })

    print(f"Selected {len(known_items)} known Q/A pairs.")

    # -------------------------------------------------------
    # 3. SAMPLE UNKNOWN QUESTIONS
    # -------------------------------------------------------
    unknown_sample = random.sample(UNKNOWN_QUESTIONS, N_UNKNOWN)

    unknown_items = []
    for idx, q in enumerate(unknown_sample):
        unknown_items.append({
            "id": f"u{idx+1}",
            "type": "unknown",
            "question": q,
            "answer": None
        })

    print(f"Added {len(unknown_items)} unknown (out-of-domain) questions.")

    # -------------------------------------------------------
    # 4. WRITE JSONL OUTPUT
    # -------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        for item in known_items + unknown_items:
            f.write(json.dumps(item) + "\n")

    print(f"\nEval set saved to: {OUTPUT_FILE}")
    print(f"Total lines: {len(known_items) + len(unknown_items)}\n")
    print("=== DONE ===\n")


if __name__ == "__main__":
    main()
