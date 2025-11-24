"""
evaluate_rag.py
---------------------------------------------------------
Automated Evaluation for RAG Pipeline

This version:
    • Loads RbcRetriever + Phi-3.5 generator
    • Mirrors FastAPI /ask logic exactly
    • Uses the new Balanced grounding logic:
          from src.generation.generator import is_grounded
    • Evaluates:
          - known questions (should give grounded answer)
          - unknown questions (should answer "I don't know.")
    • Saves detailed per-question logs
    • Prints hallucination summary

Run:
    python evaluate_rag.py \
        --eval-file data/eval/rbc_eval_set.jsonl \
        --output-file data/eval/rbc_eval_results.jsonl \
        --top-k 5
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

# Project modules
from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer, is_grounded


# ---------------------------------------------------------
# CLEAN RETRIEVAL  (copy of FastAPI main.py)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.40, max_items: int = 4):
    """Same logic as FastAPI clean_retrieval()."""
    if not results:
        return []

    sorted_results = sorted(
        results,
        key=lambda r: r.get("score", 0.0),
        reverse=True
    )

    filtered = [
        r for r in sorted_results
        if r.get("score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    chunks = [r["chunk"].strip() for r in filtered]
    return chunks[:max_items]


# ---------------------------------------------------------
# TOKENIZATION UTILS (kept for metrics)
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


# ---------------------------------------------------------
# EVALUATE ONE QUESTION
# ---------------------------------------------------------
def evaluate_one(example: dict, retriever: RbcRetriever, top_k: int):
    q_id = example.get("id")
    question = example.get("question")
    gold_answer = example.get("answer")  # None for unknown
    q_type = example.get("type", "unknown")

    # ---- 1. RETRIEVAL ----
    retrieved = retriever.search(question, top_k=top_k)
    clean_chunks = clean_retrieval(retrieved)

    # ---- 2. GENERATION (UNPACK ANSWER STRING) ----
    rag_answer, _ = generate_answer(question, clean_chunks)

    # ---- 3. METRICS (for debugging, not for decision)
    gold_tokens = tokenize(gold_answer) if gold_answer else []
    ans_tokens = tokenize(rag_answer)
    ctx_tokens = tokenize(" ".join(clean_chunks))

    context_overlap = jaccard(gold_tokens, ctx_tokens) if gold_tokens else 0.0
    grounding_score = jaccard(ans_tokens, ctx_tokens) if ans_tokens else 0.0

    # ---- 4. FINAL HALLUCINATION DECISION ----
    if q_type == "unknown":
        # unknown should say "I don't know."
        hallucinated = rag_answer.lower().strip() != "i don't know."
    else:
        # known → check grounding using the new shared logic
        hallucinated = not is_grounded(rag_answer, clean_chunks)

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
# MAIN EVALUATION LOOP
# ---------------------------------------------------------
def evaluate(eval_file: Path, output_file: Path, top_k: int):
    print("Loading retriever + generator...")
    retriever = RbcRetriever()
    print("Components loaded.\n")

    # Load eval set
    examples = []
    with open(eval_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} test questions.\n")

    results = []
    for ex in examples:
        results.append(evaluate_one(ex, retriever, top_k))

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # ---- Summary ----
    known = [r for r in results if r["type"] == "known"]
    unknown = [r for r in results if r["type"] == "unknown"]

    known_hall = sum(r["hallucinated"] for r in known)
    unknown_hall = sum(r["hallucinated"] for r in unknown)

    print("=== Evaluation Summary ===")
    print(f"Total questions:      {len(results)}")
    print(f"Known questions:      {len(known)}")
    print(f"Unknown questions:    {len(unknown)}\n")

    print(f"Known hallucinations:   {known_hall} / {len(known)}")
    print(f"Unknown hallucinations: {unknown_hall} / {len(unknown)}\n")

    print(f"Results saved to: {output_file}")

    return results


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument("--eval-file", type=str, required=True,
                        help="Path to evaluation JSONL file")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Where to save evaluation results")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Retriever top_k (default=5)")

    args = parser.parse_args()

    evaluate(
        Path(args.eval_file),
        Path(args.output_file),
        args.top_k
    )
