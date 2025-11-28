"""
metrics.py
---------------------------------------------------------
Unified evaluation utilities for grounding, hallucination,
and retrieval-quality analysis.

This file now delegates all grounding logic to:
    • generator.is_grounded
    • generator.grounding_details

This ensures 100% alignment between:
    - generator.py (production FastAPI)
    - evaluate_rag.py (Phase 5 evaluation)
    - monitoring/analytics (Phase 8)

Safe for:
    - Local Phi-3.5
    - Cloud Run Groq
"""

import re
from typing import List, Dict

# Import the unified grounding logic from generator.py
from src.generation.generator import (
    is_grounded as _is_grounded_core,
    grounding_details as _grounding_details_core,
)

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

    Independent of grounding logic.
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
# NORMALIZATION UTILITIES FOR "I DON'T KNOW"
# =========================================================
def _normalize_idk(text: str) -> str:
    """
    Normalizes variants of 'I don't know' for strict evaluation.

    Example outputs:
      "I don't know."  -> "i dont know"
      "I do not know!" -> "i do not know"
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^a-z\s]", "", text.lower())
    return " ".join(cleaned.split())


# =========================================================
# UNIFIED GROUNDING — WRAPPERS AROUND GENERATOR LOGIC
# =========================================================
def is_grounded(answer: str, chunks: List[str]) -> bool:
    """
    Wrapper around generator.is_grounded.
    Ensures consistent grounding everywhere.
    """
    return _is_grounded_core(answer, chunks)


def evaluate_grounding(answer: str, chunks: List[str]) -> Dict:
    """
    Wrapper around generator.grounding_details.
    Returns:
      - grounded (bool)
      - grounding_score (0–1)
      - context_overlap (0–1)
    """
    return _grounding_details_core(answer, chunks)


# =========================================================
# HALLUCINATION DETECTION — UNIFIED LOGIC
# =========================================================
def detect_hallucination(answer: str, chunks: List[str], q_type: str) -> bool:
    """
    Unified hallucination logic matching FastAPI + evaluation.

    unknown questions:
        → Must effectively mean "I don't know."
    known questions:
        → Must be grounded in context.
    """
    norm = _normalize_idk(answer)

    if q_type == "unknown":
        return norm not in {
            "i dont know",
            "i do not know",
        }

    # Known → must be grounded according to unified logic
    return not _is_grounded_core(answer, chunks)
