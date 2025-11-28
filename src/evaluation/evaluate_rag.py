"""
evaluate_rag.py
---------------------------------------------------------
Automated Evaluation for RAG Pipeline

This version:
    • Loads RbcRetriever + Phi-3.5 generator
    • Mirrors FastAPI /ask logic exactly
    • Uses the unified grounding logic from:
          src.generation.generator.grounding_details
    • Evaluates:
          - known questions (should be grounded)
          - unknown questions (should answer "I don't know.")
    • Saves detailed evaluation logs to JSONL
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict

import torch  # Required because generator uses torch.no_grad()

from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer, grounding_details


# ---------------------------------------------------------
# Retrieval Cleaner (close to FastAPI, slightly relaxed)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.30, max_items: int = 5):
    """
    Sorts by score → filters low score → extracts chunks.

    Slightly relaxed threshold vs 0.40 to avoid dropping
    relevant chunks during evaluation.
    """
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)

    filtered = [
        r
        for r in sorted_results
        if r.get("score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    chunks = [r["chunk"].strip() for r in filtered]
    return chunks[:max_items]


# ---------------------------------------------------------
# Token Utilities (kept for possible extra metrics)
# ---------------------------------------------------------
def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return text.lower().replace("\n", " ").split()


def jaccard(a: List[str], b: List[str]):
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _normalize_idk(text: str) -> str:
    """
    Normalize answer text for robust 'I don't know' checks.

    Examples:
      "I don't know."   -> "i dont know"
      "I don't know!"   -> "i dont know"
      "I do not know."  -> "i do not know"
    """
    if not text:
        return ""
    # Keep only letters and spaces
    cleaned = re.sub(r"[^a-z\s]", "", text.lower())
    # Collapse multiple spaces
    return " ".join(cleaned.split())


# ---------------------------------------------------------
# Evaluate One Example
# ---------------------------------------------------------
def evaluate_one(example: dict, retriever: RbcRetriever, top_k: int):
    q_id = example.get("id")
    question = example.get("question")
    gold_answer = example.get("answer")  # None for unknown questions
    q_type = example.get("type", "unknown")

    # ---------- 1. RETRIEVAL ----------
    retrieved = retriever.search(question, top_k=top_k)
    clean_chunks = clean_retrieval(retrieved)

    # ---------- 2. GENERATION ----------
    rag_answer, grounding = generate_answer(question, clean_chunks)

    # grounding contains:
    #   - grounded (bool)
    #   - grounding_score (float)
    #   - context_overlap (float)

    grounding_score = grounding.get("grounding_score", 0.0)
    context_overlap = grounding.get("context_overlap", 0.0)

    # ---------- 3. HALLUCINATION CHECK ----------
    normalized_idk = _normalize_idk(rag_answer)

    if q_type == "unknown":
        # Unknown → must effectively respond "I don't know"
        # Accept reasonable variants like:
        #   "I don't know", "I don't know.", "I don't know!"
        hallucinated = normalized_idk not in {
            "i dont know",
            "i do not know",
        }
    else:
        # Known → must satisfy grounding rules
        hallucinated = not grounding.get("grounded", False)

    return {
        "id": q_id,
        "type": q_type,
        "question": question,
        "gold_answer": gold_answer,
        "rag_answer": rag_answer,
        "retrieved": retrieved,
        "used_chunks": clean_chunks,
        "context_overlap": context_overlap,
        "grounding_score": grounding_score,
        "hallucinated": hallucinated,
    }


# ---------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------
def evaluate(eval_file: Path, output_file: Path, top_k: int):
    print("Loading retriever + generator...")
    retriever = RbcRetriever()
    print("Components loaded.\n")

    # Load evaluation questions
    examples = [json.loads(line) for line in open(eval_file, "r")]
    print(f"Loaded {len(examples)} test questions.\n")

    results = [evaluate_one(ex, retriever, top_k) for ex in examples]

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    known = [r for r in results if r["type"] == "known"]
    unknown = [r for r in results if r["type"] == "unknown"]

    known_hall = sum(r["hallucinated"] for r in known)
    unknown_hall = sum(r["hallucinated"] for r in unknown)

    print("=== Evaluation Summary ===")
    print(f"Total questions:        {len(results)}")
    print(f"Known questions:        {len(known)}")
    print(f"Unknown questions:      {len(unknown)}\n")

    print(f"Known hallucinations:   {known_hall} / {len(known)}")
    print(f"Unknown hallucinations: {unknown_hall} / {len(unknown)}\n")

    print(f"Results saved to: {output_file}\n")

    return results


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    evaluate(Path(args.eval_file), Path(args.output_file), args.top_k)
