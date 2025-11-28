"""
metrics.py
---------------------------------------------------------
Unified evaluation utilities for grounding, hallucination,
and retrieval-quality analysis.

100% aligned with:
    - generator.py   (is_grounded, grounding_details)
    - evaluate_rag.py
    - FastAPI backend /ask
    - Phase 8 monitoring dashboards

This ensures:
    • No evaluation/back-end drift
    • No duplicate grounding logic
    • Strict literal-mode behavior everywhere
"""

import re
from typing import List, Dict

# Import PRODUCTION grounding logic directly
from src.generation.generator import (
    is_grounded as _is_grounded_core,
    grounding_details as _grounding_details_core,
)

# =========================================================
# TOKENIZATION UTILS (for retrieval-hit metric only)
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
# SIMPLE RETRIEVAL-HIT METRIC
# (Independent of grounding logic)
# =========================================================

def evaluate_retrieval_hit(question: str, retrieved_chunks: List[str], gold_answer: str) -> float:
    """
    Token-overlap between gold answer and retrieved context.
    Used only as a light diagnostic for retrieval quality.
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
# NORMALIZED “I DON’T KNOW” UTILITY
# (Used by evaluation + monitoring)
# =========================================================

def _normalize_idk(text: str) -> str:
    """
    Robust normalization for IDK variants:
        "I don't know."
        "I don’t know"
        "I do not know!"
        "i dont KNOW??"
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^a-z\s]", "", text.lower())
    return " ".join(cleaned.split())


# =========================================================
# UNIFIED GROUNDING — DIRECT DELEGATION
# =========================================================

def is_grounded(answer: str, chunks: List[str]) -> bool:
    """
    Delegates all grounding logic to generator.is_grounded.
    NO evaluation-specific logic here.
    """
    return _is_grounded_core(answer, chunks)


def evaluate_grounding(answer: str, chunks: List[str]) -> Dict:
    """
    Delegates to generator.grounding_details.
    Returns:
        grounded (bool)
        grounding_score (0–1)
        context_overlap (0–1)
    """
    return _grounding_details_core(answer, chunks)


# =========================================================
# UNIFIED HALLUCINATION LOGIC
# EXACTLY matches evaluate_rag.py + FastAPI
# =========================================================

def detect_hallucination(answer: str, chunks: List[str], q_type: str) -> bool:
    """
    For consistency across:
        • /ask endpoint
        • evaluate_rag.py
        • Phase 8 monitoring

    unknown questions:
        MUST effectively answer “I don’t know.”

    known questions:
        MUST satisfy grounding rules.
    """
    norm = _normalize_idk(answer)

    if q_type == "unknown":
        return norm not in {
            "i dont know",
            "i do not know",
        }

    # Known → must be grounded
    return not _is_grounded_core(answer, chunks)
