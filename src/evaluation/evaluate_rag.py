"""
evaluate_rag.py
---------------------------------------------------------
Phase 5 Evaluation — Strict Literal Mode RAG

This version:
    • Uses the SAME retrieval logic as the FastAPI backend
    • Uses the SAME generator logic (strict literal mode)
    • Uses grounding_details() from generator.py
    • Evaluates:
        - known questions  → MUST be grounded
        - unknown questions → MUST say "I don't know."
    • Produces JSONL evaluation logs identical to Phase 5
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict

import torch  # Needed because generator uses torch.no_grad()

from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer, grounding_details


# ---------------------------------------------------------
# Retrieval Cleaner (matches new main.py)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.32, max_items: int = 4):
    """
    EXACT MATCH with FastAPI clean_retrieval()

    Sorting by final_score → returning ONLY chunks
    (no citation IDs because generator assigns CIT:1…)
    """
    if not results:
        return []

    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    strong = [
        r for r in ordered
        if r.get("final_score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    chunks = [r["chunk"].strip() for r in strong][:max_items]
    return chunks


# ---------------------------------------------------------
# IDK Normalizer
# ---------------------------------------------------------
def _normalize_idk(text: str) -> str:
    """
    Normalize model outputs to detect variants like:
    - "I don't know"
    - "I don’t know."
    - "I do not know"
    - "I don't know!"
    """
    if not text:
        return ""

    cleaned = re.sub(r"[^a-z\s]", "", text.lower())
    return " ".join(cleaned.split())


# ---------------------------------------------------------
# Evaluate One Example
# ---------------------------------------------------------
def evaluate_one(example: dict, retriever: RbcRetriever, top_k: int):
    q_id = example.get("id")
    question = example.get("question")
    gold_answer = example.get("answer")  # None for unknown
    q_type = example.get("type", "unknown")

    # ---------- 1. RETRIEVAL ----------
    retrieved = retriever.search(question, top_k=top_k)
    clean_chunks = clean_retrieval(retrieved)

    # ---------- 2. GENERATION ----------
    rag_answer, grounding = generate_answer(question, clean_chunks)

    grounding_score = grounding.get("grounding_score", 0.0)
    context_overlap = grounding.get("context_overlap", 0.0)
    grounded_flag = grounding.get("grounded", False)

    # ---------- 3. HALLUCINATION CHECK ----------
    normalized_idk = _normalize_idk(rag_answer)

    if q_type == "unknown":
        # Must answer IDK
        hallucinated = normalized_idk not in {
            "i dont know",
            "i do not know",
        }
    else:
        # Known → MUST be grounded
        hallucinated = not grounded_flag

    return {
        "id": q_id,
        "type": q_type,
        "question": question,
        "gold_answer": gold_answer,
        "rag_answer": rag_answer,
        "retrieved": retrieved,
        "used_chunks": clean_chunks,
        "grounding_score": grounding_score,
        "context_overlap": context_overlap,
        "hallucinated": hallucinated,
    }


# ---------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------
def evaluate(eval_file: Path, output_file: Path, top_k: int):
    print("Loading retriever + generator...")
    retriever = RbcRetriever()
    print("Components loaded.\n")

    # Load evaluation set
    examples = [json.loads(line) for line in open(eval_file, "r")]
    print(f"Loaded {len(examples)} test questions.\n")

    results = [evaluate_one(ex, retriever, top_k) for ex in examples]

    # Save JSONL output
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
