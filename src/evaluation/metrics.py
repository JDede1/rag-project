"""
metrics.py
---------------------------------------------------------
Shared evaluation utilities for grounding, hallucination,
and retrieval-quality analysis.

This file mirrors:
    • generator.py grounding logic
    • FastAPI hallucination checks
    • RAG evaluation in Phase 5

All functions here are safe for both:
    - Local (Phi-3.5)
    - Cloud Run (Groq)
"""

import re
from typing import List, Dict


# =========================================================
# TOKENIZATION UTILS
# =========================================================

STOPWORDS = {
    "the", "is", "a", "to", "of", "and", "in", "for", "on",
    "by", "you", "your", "or", "we", "with", "at", "from",
    "as", "an", "it", "be",
}

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


# =========================================================
# SIMPLE METRIC: RETRIEVAL HIT
# =========================================================

def evaluate_retrieval_hit(question: str, retrieved_chunks: List[str], gold_answer: str) -> float:
    """
    Computes token overlap between gold answer and retrieved context.
    Only used during evaluation, not during generation.
    """
    if not gold_answer or not retrieved_chunks:
        return 0.0

    gold_tokens = tokenize(gold_answer)
    ctx_tokens = tokenize(" ".join(retrieved_chunks))

    if not gold_tokens or not ctx_tokens:
        return 0.0

    intersection = set(gold_tokens) & set(ctx_tokens)
    union = set(gold_tokens) | set(ctx_tokens)

    return len(intersection) / max(1, len(union))


# =========================================================
# GROUNDING LOGIC (MIRRORS generator.py)
# =========================================================

_PHONE_RE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")

def is_grounded(answer: str, chunks: List[str]) -> bool:
    """
    Determines if a model answer strictly comes from the retrieved context.
    Mirrors generator.py to maintain consistency.
    """
    if not answer:
        return False

    ans = answer.lower().strip()

    # Allow "I don't know."
    if ans in {"i don't know", "i don't know."}:
        return True

    if not chunks:
        return False

    ctx_text = " ".join(chunks).lower()

    # Exact snippet match
    if ans.rstrip(".") in ctx_text:
        return True

    # Phone numbers
    if set(_PHONE_RE.findall(ans)) & set(_PHONE_RE.findall(ctx_text)):
        return True

    # Numbers (percentages, fees, amounts)
    if set(_NUMBER_RE.findall(ans)) & set(_NUMBER_RE.findall(ctx_text)):
        return True

    # Token overlap
    ans_tokens = tokenize(ans)
    ctx_tokens = set(tokenize(ctx_text))

    overlap = [t for t in ans_tokens if t in ctx_tokens]

    # Require minimum overlap
    return len(overlap) >= 2


def evaluate_grounding(answer: str, chunks: List[str]) -> Dict:
    """
    Computes:
      - grounded flag
      - grounding score (0–1)
      - context token overlap ratio
    """
    if not answer or not chunks:
        return {
            "grounded": False,
            "grounding_score": 0.0,
            "context_overlap": 0.0,
        }

    ctx_text = " ".join(chunks).lower()
    ans_text = answer.lower().rstrip(".")

    ans_tokens = tokenize(ans_text)
    ctx_tokens = set(tokenize(ctx_text))

    if not ans_tokens:
        return {
            "grounded": False,
            "grounding_score": 0.0,
            "context_overlap": 0.0,
        }

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap) / max(1, len(ans_tokens))

    grounded_flag = is_grounded(answer, chunks)

    # Weighted score
    grounding_score = (0.7 * int(grounded_flag)) + (0.3 * overlap_ratio)

    return {
        "grounded": grounded_flag,
        "grounding_score": round(float(grounding_score), 4),
        "context_overlap": round(float(overlap_ratio), 4),
    }


# =========================================================
# HALLUCINATION DETECTION
# =========================================================

def detect_hallucination(answer: str, chunks: List[str], q_type: str) -> bool:
    """
    Final hallucination decision used in evaluation.

    unknown questions:
        - MUST answer "I don't know."
    known questions:
        - answer must be grounded in context
    """
    ans = answer.lower().strip()

    if q_type == "unknown":
        return ans not in {"i don't know", "i don't know."}

    # Known question → Check grounding
    return not is_grounded(answer, chunks)
