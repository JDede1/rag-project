"""
evaluate_rag.py 
=================================================================

Matches backend EXACTLY:
    • clean_retrieval()
    • focus_context()
    • strict-literal generation
    • grounding_details()
    • is_grounded()
    • hallucination detection

This ensures Phase 5 = backend behavioral parity.
"""

import json
import re
import argparse
from pathlib import Path

import torch

# Production imports
from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import (
    generate_answer,
    grounding_details,
    is_grounded,
)


# ---------------------------------------------------------
# CLEAN RETRIEVAL (must match backend)
# ---------------------------------------------------------
def clean_retrieval(results, score_threshold=0.32, max_items=4):
    if not results:
        return []

    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    strong = [
        r for r in ordered
        if r.get("final_score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    return [r["chunk"].strip() for r in strong][:max_items]


# ---------------------------------------------------------
# CONTEXT FOCUS (THE IMPORTANT FIX)
# ---------------------------------------------------------
def focus_context(query: str, chunks: list) -> list:
    """Matches backend focus_context logic EXACTLY."""
    if not chunks:
        return chunks

    q = query.lower()

    # Lost / stolen
    if "lost" in q or "stolen" in q:
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in [
                "lost", "stolen", "permanently lost", "misplaced"
            ])
        ]
        if topical:
            return topical

    # Fraud / unauthorized / dispute
    if any(k in q for k in ["fraud", "unauthorized", "dispute"]):
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["fraud", "unauthorized", "dispute"])
        ]
        if topical:
            return topical

    # Interac / e-transfer
    if any(k in q for k in ["interac", "e-transfer", "etransfer", "transfer"]):
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["interac", "e-transfer", "etransfer", "transfer"])
        ]
        if topical:
            return topical

    # Password / login / reset
    if any(k in q for k in ["password", "login", "reset"]):
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["password", "login", "reset", "passcode"])
        ]
        if topical:
            return topical

    return chunks


# ---------------------------------------------------------
# Normalize "I don't know"
# ---------------------------------------------------------
def _normalize_idk(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"[^a-z\s]", "", text.lower())
    return " ".join(cleaned.split())


# ---------------------------------------------------------
# Evaluate a single example  (STRICT)
# ---------------------------------------------------------
def evaluate_one(example, retriever, top_k):
    qid = example["id"]
    question = example["question"]
    gold = example.get("answer")
    q_type = example.get("type", "known")

    # 1) Retrieval
    retrieved = retriever.search(question, top_k=top_k)
    clean = clean_retrieval(retrieved)

    # 🔥 APPLY FOCUS CONTEXT (backend parity)
    clean = focus_context(question, clean)

    # 2) Generator (strict-literal)
    answer, grounding = generate_answer(question, clean)

    gs = grounding.get("grounding_score", 0.0)
    overlap = grounding.get("context_overlap", 0.0)

    # 3) Hallucination logic
    norm = _normalize_idk(answer)

    if q_type == "unknown":
        hallucinated = norm not in {"i dont know", "i do not know"}
    else:
        hallucinated = not is_grounded(answer, clean)

    return {
        "id": qid,
        "type": q_type,
        "question": question,
        "gold_answer": gold,
        "rag_answer": answer,
        "retrieved": retrieved,
        "used_chunks": clean,
        "grounding_score": gs,
        "context_overlap": overlap,
        "hallucinated": hallucinated,
    }


# ---------------------------------------------------------
# MAIN EVALUATION LOOP
# ---------------------------------------------------------
def evaluate(eval_file: Path, output_file: Path, top_k: int):
    print("Loading retriever…")
    retriever = RbcRetriever()
    print("Loaded.\n")

    examples = [json.loads(line) for line in open(eval_file, "r")]
    print(f"Loaded {len(examples)} examples.\n")

    results = [evaluate_one(ex, retriever, top_k) for ex in examples]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    known = [r for r in results if r["type"] == "known"]
    unknown = [r for r in results if r["type"] == "unknown"]

    print("=== SUMMARY ===")
    print(f"Known hallucinations:   {sum(r['hallucinated'] for r in known)} / {len(known)}")
    print(f"Unknown hallucinations: {sum(r['hallucinated'] for r in unknown)} / {len(unknown)}")
    print(f"Saved → {output_file}\n")

    return results


# ---------------------------------------------------------
# CLI entry
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    evaluate(Path(args.eval_file), Path(args.output_file), args.top_k)
