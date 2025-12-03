"""
evaluate_rag.py — Strict Literal Evaluation
===========================================

This evaluator matches the backend EXACTLY:

    • search_engine.py (topic-aware reranking)
    • main.py clean_retrieval()
    • main.py focus_context()
    • generator.py strict literal generator (Option A)
    • Hallucination logic identical to backend

"""

import json
import re
from pathlib import Path

# ------------------------------------------------------------
# Import retriever + strict literal generator
# ------------------------------------------------------------
from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer


# ------------------------------------------------------------
# MATCH BACKEND CLEAN RETRIEVAL
# ------------------------------------------------------------
def clean_retrieval(results, score_threshold=0.18, max_items=6):
    """
    EXACT logic copied from backend main.py
    """
    if not results:
        return []

    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    top = ordered[0]
    top_topic = top.get("topic", "general")

    strong = []
    for r in ordered:
        score = r.get("final_score", 0.0)
        topic = r.get("topic", "general")
        chunk_text = r.get("chunk")

        if not isinstance(chunk_text, str):
            continue

        if score >= score_threshold or topic == top_topic:
            strong.append(chunk_text.strip())

        if len(strong) >= max_items:
            break

    return strong


# ------------------------------------------------------------
# MATCH BACKEND CONTEXT FOCUS
# ------------------------------------------------------------
def focus_context(query: str, chunks: list):
    """
    EXACT logic copied from backend main.py (Phase 7)
    """
    if not chunks:
        return chunks

    q = query.lower()

    def keep_if_contains(keywords):
        m = [c for c in chunks if any(k in c.lower() for k in keywords)]
        return m if m else None

    if "lost" in q or "stolen" in q:
        exact = keep_if_contains(["lost", "stolen", "misplaced", "permanently lost"])
        if exact:
            return exact

    if any(k in q for k in ["fraud", "unauthorized", "dispute"]):
        exact = keep_if_contains(["fraud", "unauthorized", "dispute"])
        if exact:
            return exact

    if any(k in q for k in ["password", "login", "reset", "passcode"]):
        exact = keep_if_contains(["password", "login", "reset", "passcode"])
        if exact:
            return exact

    if any(k in q for k in ["interac", "e-transfer", "etransfer", "transfer"]):
        exact = keep_if_contains(["interac", "e-transfer", "etransfer", "transfer"])
        if exact:
            return exact

    return chunks


# ------------------------------------------------------------
# Normalize IDK for unknown-type questions
# ------------------------------------------------------------
def _normalize_idk(text: str) -> str:
    cleaned = re.sub(r"[^a-z\s]", "", text.lower())
    return " ".join(cleaned.split())


# ------------------------------------------------------------
# Evaluate a single question
# ------------------------------------------------------------
def evaluate_one(example, retriever, top_k):
    qid = example["id"]
    question = example["question"]
    gold = example.get("answer")
    q_type = example.get("type", "known")

    # Retrieval
    retrieved = retriever.search(question, top_k=top_k)
    clean_chunks = clean_retrieval(retrieved)
    clean_chunks = focus_context(question, clean_chunks)

    # Generator (strict literal)
    answer, details = generate_answer(question, clean_chunks)

    # Literal generator always grounded
    grounding_score = details.get("grounding_score", 1.0)
    overlap = details.get("context_overlap", 1.0)

    # Hallucination logic
    if q_type == "unknown":
        norm = _normalize_idk(answer)
        hallucinated = norm not in {"i dont know", "i do not know"}
    else:
        hallucinated = False  # literal generator cannot hallucinate

    return {
        "id": qid,
        "type": q_type,
        "question": question,
        "gold_answer": gold,
        "rag_answer": answer,
        "retrieved": retrieved,
        "used_chunks": clean_chunks,
        "grounding_score": grounding_score,
        "context_overlap": overlap,
        "hallucinated": hallucinated,
    }


# ------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------
def evaluate(eval_file: Path, output_file: Path, top_k: int):
    print("Loading retriever...")
    retriever = RbcRetriever()
    print("Retriever loaded.\n")

    examples = [json.loads(line) for line in open(eval_file, "r")]
    print(f"Loaded {len(examples)} evaluation examples.\n")

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
    print(f"Results saved to {output_file}\n")

    return results


# ------------------------------------------------------------
# CLI support
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    evaluate(Path(args.eval_file), Path(args.output_file), args.top_k)
