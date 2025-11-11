"""
inspect_dataset.py
-------------------------------------
Utility for inspecting cleaned FAQ datasets.

Features:
    • Loads and displays dataset info (shape, sample QAs, length stats)
    • Optional --report flag saves a Markdown summary under data/reports/
Usage:
    python src/preprocess/inspect_dataset.py data/processed/rbc_faqs_clean.parquet
    python src/preprocess/inspect_dataset.py data/processed/rbc_faqs_clean.parquet --report
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# -------------------------
# MAIN FUNCTION
# -------------------------
def inspect_dataset(parquet_path: str, save_report: bool = False):
    """Load and inspect a cleaned FAQ dataset."""
    path = Path(parquet_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"📂 Loading dataset: {path}")
    df = pd.read_parquet(path)
    print(f"✅ Shape: {df.shape}")

    # -------------------------
    # Dataset samples
    # -------------------------
    print("\n🔹 Sample FAQs (first 5):\n")
    samples = []
    for i, row in df.head(5).iterrows():
        q, a = row['question'], row['answer']
        print(f"Q{i+1}: {q}")
        print(f"A{i+1}: {a[:300]}...\n---\n")
        samples.append((q, a[:300]))

    # -------------------------
    # Statistics
    # -------------------------
    q_len_avg = df["question"].str.len().mean()
    a_len_avg = df["answer"].str.len().mean()
    missing = df.isna().sum()
    has_missing = missing.any()

    print("📊 Basic text statistics:")
    print(f"- Avg question length: {q_len_avg:.2f} characters")
    print(f"- Avg answer length:   {a_len_avg:.2f} characters")

    if has_missing:
        print("\n⚠️ Missing values detected:")
        print(missing[missing > 0])
    else:
        print("\n✅ No missing values detected.")

    print("\n✨ Inspection complete.\n")

    # -------------------------
    # Optional: Save Markdown Report
    # -------------------------
    if save_report:
        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = reports_dir / f"{path.stem}_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Dataset Inspection Report — {path.name}\n\n")
            f.write(f"🕒 Generated on: {timestamp}\n\n")
            f.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns\n\n")
            f.write("## 📊 Text Statistics\n")
            f.write(f"- Avg question length: {q_len_avg:.2f} characters\n")
            f.write(f"- Avg answer length: {a_len_avg:.2f} characters\n")
            f.write(f"- Missing values: {'Yes' if has_missing else 'No'}\n\n")

            f.write("## 🔹 Sample FAQs\n")
            for idx, (q, a) in enumerate(samples, 1):
                f.write(f"**Q{idx}:** {q}\n\n")
                f.write(f"**A{idx}:** {a}...\n\n---\n\n")

        print(f"📝 Markdown report saved → {report_path}")


# -------------------------
# CLI ENTRY POINT
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/preprocess/inspect_dataset.py <path_to_parquet> [--report]")
        sys.exit(1)

    parquet_path = sys.argv[1]
    save_report = "--report" in sys.argv
    inspect_dataset(parquet_path, save_report)
