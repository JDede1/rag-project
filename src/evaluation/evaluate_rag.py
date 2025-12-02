"""
evaluate_rag.py 
==========================================

This version:
    • 100% matches backend behavior
    • Includes focus_context() (critical fix)
    • Uses strict-literal generator (generate_answer)
    • Uses production grounding logic (is_grounded, grounding_details)
    • Detects hallucinations exactly like backend & monitoring
    • Produces JSONL output for Phase 5

Output:
    results/*.jsonl
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict

import torch  # required by generator during inference

# Production components
from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import (
    generate_answer,
    grounding_details,
    is_grounded,
)


# =========================================================
# CLEAN RETRIEVAL (must match backend)
# =========================================================
def clean_retrieval(results: list, score_threshold: float = 0.32, max_items: int = 4):
    """
    EXACT MATCH with backend clean_retrieval()

    Sorts retrieved chunks by final_score and keeps only
    strong, clean items.
    """
    if not results:
        return []

    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    strong = [
        r for r in ordered
        if r.get("final_score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    return [r["chunk"].strip() for r in strong][:max_items]


# =========================================================
# TOPIC FOCUS (critical fix — must match backend EXACTLY)
# =========================================================
def focus_context(query: str, chunks: list) -> list:
    """
    Same logic as backend main.py.

    Keeps only context relevant to the dominant intent:
        - lost/stolen
        - fraud/dispute
        - interac/e-transfer
        - password/login/reset
    """
    if not chunks:
        return chunks

    q = query.lower()

    # LOST / STOLEN
    if "lost" in q or "stolen" in q:
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["lost", "stolen", "permanently lost", "misplaced"])
        ]
        if topical:
            return topical

    # FRAUD / unauthorized / dispute
    if any(k in q for k in ["fraud", "unauthorized", "dispute"]):
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["fraud", "unauthorized", "dispute"])
        ]
        if topical:
            return topical

    # INTERAC / TRANSFER
    if any(k in q for k in ["interac", "e-transfer", "etransfer", "e transfer", "transfer"]):
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["interac", "transfer", "etransfer", "e-transfer"])
        ]
        if topical:
            return topical

    # PASSWORD / LOGIN / RESET
    if any(k in q for k in ["password", "login", "reset"]):
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["password", "login", "reset", "passcode"])
        ]
        if topical:
            return topical

    # Default fallback
    return chunks


# =========================================================
# NORMALIZING "I DON'T KNOW"
# =========================================================
def _normalize_idk(text: str) -> str:
    """
    Normalize variations of 'I don't know' so evaluation
    detects IDK reliably.
    """
    if not text:
        return ""

    cleaned = re.sub(r"[^a-z\s]", "", text.lower())
    return " ".join(cleaned.split())


# =========================================================
# EVALUATE ONE QUESTION
# =========================================================
def evaluate_one(example: dict, retriever: RbcRetriever, top_k: int):
    q_id = example.get("id")
    question = example.get("question")
    gold_answer = example.get("answer")  # None for unknown
    q_type = example.get("type", "known")

    # -----------------------------------------------------
    # 1. RETRIEVAL
    # -----------------------------------------------------
    retrieved = retriever.search(question, top_k=top_k)

    clean_chunks = clean_retrieval(retrieved)

    # CRITICAL: match backend EXACTLY
    clean_chunks = focus_context(question, clean_chunks)

    # -----------------------------------------------------
    # 2. GENERATION (strict literal)
    # -----------------------------------------------------
    rag_answer, grounding = generate_answer(question, clean_chunks)

    grounding_score = grounding.get("grounding_score", 0.0)
    context_overlap = grounding.get("context_overlap", 0.0)

    # -----------------------------------------------------
    # 3. HALLUCINATION (must match backend logic)
    # -----------------------------------------------------
    normalized_idk = _normalize_idk(rag_answer)

    if q_type == "unknown":
        hallucinated = normalized_idk not in {"i dont know", "i do not know"}
    else:
        # known → must be grounded
        hallucinated = not is_grounded(rag_answer, clean_chunks)

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


# =========================================================
# MAIN LOOP
# =========================================================
def evaluate(eval_file: Path, output_file: Path, top_k: int):
    print("Loading retriever & generator…")
    retriever = RbcRetriever()
    print("Components ready.\n")

    examples = [json.loads(line) for line in open(eval_file, "r")]
    print(f"Loaded {len(examples)} test questions.\n")

    results = [evaluate_one(ex, retriever, top_k) for ex in examples]

    # save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # summary
    known = [r for r in results if r["type"] == "known"]
    unknown = [r for r in results if r["type"] == "unknown"]

    known_h = sum(r["hallucinated"] for r in known)
    unknown_h = sum(r["hallucinated"] for r in unknown)

    print("=== Evaluation Summary ===")
    print(f"Total questions:        {len(results)}")
    print(f"Known questions:        {len(known)}")
    print(f"Unknown questions:      {len(unknown)}\n")
    print(f"Known hallucinations:   {known_h} / {len(known)}")
    print(f"Unknown hallucinations: {unknown_h} / {len(unknown)}\n")
    print(f"Saved to: {output_file}\n")

    return results


# =========================================================
# CLI ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    evaluate(Path(args.eval_file), Path(args.output_file), args.top_k)
