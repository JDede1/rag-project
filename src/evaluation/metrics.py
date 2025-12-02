"""
metrics.py
---------------------------------------------------------
Unified evaluation utilities for grounding, hallucination,
and retrieval-quality analysis.

100% aligned with:
    - generator.py   (is_grounded, grounding_details)
    - evaluate_rag.py
    - FastAPI backend /ask
    - Monitoring dashboards
"""

import re
from typing import List, Dict

# Import PRODUCTION grounding logic directly
from src.generation.generator import (
    is_grounded as _is_grounded_core,
    grounding_details as _grounding_details_core,
)

# =========================================================
# TOKENIZATION UTILS (retrieval-hit diagnostic only)
# =========================================================

STOPWORDS = {
    "the", "is", "a", "to", "of", "and", "in", "for", "on",
    "by", "you", "your", "or", "we", "with", "at", "from",
    "as", "an", "it", "be",
}

def tokenize(text: str) -> List[str]:
    """
    Light tokenization for retrieval-hit diagnostics.
    This does NOT affect grounding or hallucination detection.
    """
    if not text:
        return []
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


# =========================================================
# SIMPLE RETRIEVAL-HIT METRIC (DIAGNOSTIC ONLY)
# =========================================================

def evaluate_retrieval_hit(question: str, retrieved_chunks: List[str], gold_answer: str) -> float:
    """
    Computes token overlap between gold answer and retrieved context.
    Purely diagnostic → not used for grounding or hallucinations.
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
# NORMALIZED “I DON'T KNOW” UTILITY
# =========================================================

def _normalize_idk(text: str) -> str:
    """
    Robust normalization to detect all variants of “I don't know”:

        "I don't know."
        "I don’t know"
        "I do not know!"
        "i dont KNOW??"

    Used by:
        - Phase 5 evaluation
        - Phase 8 monitoring
        - Unified backend logic
    """
    if not text:
        return ""

    cleaned = re.sub(r"[^a-z\s]", "", text.lower())  # remove punctuation
    return " ".join(cleaned.split())  # normalize whitespace


# =========================================================
# UNIFIED GROUNDING (delegates to generator.py)
# =========================================================

def is_grounded(answer: str, chunks: List[str]) -> bool:
    """
    Delegates to generator.is_grounded() so behavior is identical
    across backend, evaluation, and monitoring.
    """
    return _is_grounded_core(answer, chunks)


def evaluate_grounding(answer: str, chunks: List[str]) -> Dict:
    """
    Delegates to generator.grounding_details().
    Returns:
        {
            "grounded": bool,
            "grounding_score": float,
            "context_overlap": float
        }
    """
    return _grounding_details_core(answer, chunks)


# =========================================================
# UNIFIED HALLUCINATION CHECK (strict literal mode)
# EXACT MATCH with:
#    - FastAPI /ask
#    - evaluate_rag.py
#    - generator.py
# =========================================================

def detect_hallucination(answer: str, chunks: List[str], q_type: str) -> bool:
    """
    unknown:
        MUST answer:   "I don't know."

    known:
        MUST be grounded (strict literal mode).

    Uses generator.is_grounded() to prevent any drift.
    """
    norm = _normalize_idk(answer)

    if q_type == "unknown":
        return norm not in {
            "i dont know",
            "i do not know",
        }

    # known → must be grounded
    return not _is_grounded_core(answer, chunks)
