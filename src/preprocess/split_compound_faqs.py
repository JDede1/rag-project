"""
split_compound_faqs.py
-------------------------------------
Detects and splits compound FAQ answers that contain multiple question-answer pairs
into separate, atomic entries suitable for embedding.

Input:
    data/processed/rbc_faqs_clean.parquet
Output:
    data/processed/rbc_faqs_refined.parquet
"""

import re
import pandas as pd
from pathlib import Path


# -------------------------
# PATH CONFIG
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_clean.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "rbc_faqs_refined.parquet"


# -------------------------
# SPLITTING LOGIC
# -------------------------
def split_compound_answer(row):
    """
    Identify and split compound Q&A sections where one 'answer' contains multiple sub-questions.

    Example:
    "Is there a number I can call...? Yes, you can call... How many corporate cards...?"
    → becomes two separate entries.
    """
    question = row["question"].strip()
    answer = row["answer"].strip()

    # If the answer has more than one question mark, we’ll try to split it
    if answer.count("?") > 1:
        # Split by question marks followed by a capital letter (likely start of a new question)
        segments = re.split(r"(?<=\?)\s+(?=[A-Z])", answer)

        sub_faqs = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue

            # Try to separate sub-question from its answer
            # Pattern: "Q? A..."
            match = re.match(r"^(.*?\?)\s*(.*)$", seg)
            if match:
                sub_q, sub_a = match.groups()
                if len(sub_q) > 10 and len(sub_a) > 10:
                    sub_faqs.append({"question": sub_q, "answer": sub_a})
            else:
                # If no clear sub-question, treat as continuation of previous answer
                sub_faqs.append({"question": question, "answer": seg})

        return sub_faqs

    # Otherwise, return as-is
    return [{"question": question, "answer": answer}]


# -------------------------
# MAIN PIPELINE
# -------------------------
def refine_faqs():
    print(f"📂 Loading {INPUT_PATH.name}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} records")

    refined_rows = []
    for _, row in df.iterrows():
        refined_rows.extend(split_compound_answer(row))

    refined_df = pd.DataFrame(refined_rows).drop_duplicates(subset=["question", "answer"])
    refined_df["question"] = refined_df["question"].str.strip()
    refined_df["answer"] = refined_df["answer"].str.strip()

    print(f"✅ Refined to {len(refined_df)} atomic FAQ entries")
    refined_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"💾 Saved refined dataset → {OUTPUT_PATH}")


if __name__ == "__main__":
    refine_faqs()
