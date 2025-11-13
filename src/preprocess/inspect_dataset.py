"""
inspect_dataset.py
-------------------------------------------------------
General-purpose inspection tool for FAQ datasets.

Supports two dataset types:

1. Q/A datasets:
       columns: question, answer

2. Chunked datasets:
       columns: question, chunk

Automatically detects which schema is present and adjusts the
inspection logic accordingly.

Features:
    • Displays shape and sample records
    • Computes text length statistics
    • Optionally saves a Markdown report

Usage:
    python src/preprocess/inspect_dataset.py data/processed/rbc_faqs_clean.parquet
    python src/preprocess/inspect_dataset.py data/processed/rbc_faq_chunks.parquet --report
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime


# -------------------------------------------------------
# Main Function
# -------------------------------------------------------
def inspect_dataset(parquet_path: str, save_report: bool = False):
    path = Path(parquet_path)
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Loading dataset: {path}")
    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")

    # ---------------------------------------------------
    # Detect dataset type
    # ---------------------------------------------------
    is_qa = {"question", "answer"}.issubset(df.columns)
    is_chunk = {"question", "chunk"}.issubset(df.columns)

    if not (is_qa or is_chunk):
        raise ValueError(
            "Dataset must contain either:\n"
            "  • question + answer columns, or\n"
            "  • question + chunk columns."
        )

    dataset_type = "Q/A" if is_qa else "Chunked"
    print(f"Detected dataset type: {dataset_type}")

    # ---------------------------------------------------
    # Display sample records
    # ---------------------------------------------------
    print("\nSample entries (first 5):\n")
    samples = []

    for i, row in df.head(5).iterrows():
        q = row["question"]
        a = row["answer"] if is_qa else row["chunk"]

        print(f"Q{i+1}: {q}")
        print(f"{'A' if is_qa else 'Chunk'}{i+1}: {a[:300]}...\n---\n")
        samples.append((q, a[:300]))

    # ---------------------------------------------------
    # Compute statistics
    # ---------------------------------------------------
    q_len_avg = df["question"].str.len().mean()
    a_col = "answer" if is_qa else "chunk"
    a_len_avg = df[a_col].str.len().mean()

    missing = df.isna().sum()
    has_missing = missing.any()

    print("Basic text statistics:")
    print(f"- Avg question length: {q_len_avg:.2f} characters")
    print(f"- Avg {a_col} length: {a_len_avg:.2f} characters")

    if has_missing:
        print("\nMissing values detected:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values detected.")

    print("\nInspection complete.\n")

    # ---------------------------------------------------
    # Optional Markdown report
    # ---------------------------------------------------
    if save_report:
        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = reports_dir / f"{path.stem}_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Dataset Inspection Report — {path.name}\n\n")
            f.write(f"Generated on: {timestamp}\n\n")
            f.write(f"**Dataset type:** {dataset_type}\n\n")
            f.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n\n")
            f.write("## Text Statistics\n")
            f.write(f"- Avg question length: {q_len_avg:.2f} characters\n")
            f.write(f"- Avg {a_col} length: {a_len_avg:.2f} characters\n")
            f.write(f"- Missing values: {'Yes' if has_missing else 'No'}\n\n")

            f.write("## Sample Entries\n")
            for idx, (q, a) in enumerate(samples, 1):
                f.write(f"**Q{idx}:** {q}\n\n")
                f.write(f"**{'A' if is_qa else 'Chunk'}{idx}:** {a}...\n\n---\n\n")

        print(f"Markdown report saved to: {report_path}")


# -------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/preprocess/inspect_dataset.py <path_to_parquet> [--report]")
        sys.exit(1)

    parquet_path = sys.argv[1]
    save_report = "--report" in sys.argv
    inspect_dataset(parquet_path, save_report)
