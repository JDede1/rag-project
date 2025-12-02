"""
metrics.py  
---------------------------------------------------------
Provides consistent evaluation utilities for:

    • Grounding (delegated to generator.py)
    • Hallucination detection (strict literal)
    • Retrieval diagnostics (token-overlap)
    • IDK normalization

Fully aligned with:
    - generator.py   
    - evaluate_rag.py
    - FastAPI backend /ask
    - Monitoring dashboards (Phase-8+)
"""

import re
from typing import List, Dict

# Import PRODUCTION grounding logic directly
from src.generation.generator import (
    is_grounded as _is_grounded_core,
    grounding_details as _grounding_details_core,
)


# =========================================================
# TOKENIZATION UTILS (retrieval diagnostics only)
# =========================================================

STOPWORDS = {
    "the", "is", "a", "to", "of", "and", "in", "for", "on",
    "by", "you", "your", "or", "we", "with", "at", "from",
    "as", "an", "it", "be",
}

def tokenize(text: str) -> List[str]:
    """
    Light tokenization for retrieval diagnostics.
    This does NOT affect grounding or hallucination detection.
    """
    if not text:
        return []
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


# =========================================================
# RETRIEVAL-HIT METRIC (diagnostic only)
# =========================================================

def evaluate_retrieval_hit(question: str, retrieved_chunks: List[str], gold_answer: str) -> float:
    """
    Measures token overlap between gold answer and retrieved chunks.
    Purely diagnostic → not used for grounding or hallucination checks.
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
# IDK NORMALIZATION 
# =========================================================

def _normalize_idk(text: str) -> str:
    """
    Robust normalization to detect all variants of “I don't know”:

        "I don't know."
        "I dont know"
        "I do not know!"
        "i dont KNOW??"

    Used by:
        - Phase 5 evaluation
        - Phase 8 monitoring
        - Unified backend logic
        - Generator fallback logic
    """
    if not text:
        return ""

    cleaned = re.sub(r"[^a-z\s]", "", text.lower())
    return " ".join(cleaned.split())


# =========================================================
# UNIFIED GROUNDING (Phase-7)
# =========================================================

def is_grounded(answer: str, chunks: List[str]) -> bool:
    """
    Delegates to generator.is_grounded().
    Ensures backend, evaluation, and metrics behave identically.
    """
    return _is_grounded_core(answer, chunks)


def evaluate_grounding(answer: str, chunks: List[str]) -> Dict:
    """
    Returns generator.grounding_details() EXACTLY.
    """
    return _grounding_details_core(answer, chunks)


# =========================================================
# UNIFIED HALLUCINATION DETECTION 
# =========================================================

def detect_hallucination(answer: str, chunks: List[str], q_type: str) -> bool:
    """
    Phase-7 Hallucination Rules:

    unknown:
        MUST answer: "I don't know."

    known:
        MUST be grounded.
        (Generator ensures strict literal fallback if ungrounded.)

    This matches:
        - FastAPI /ask
        - evaluate_rag.py
        - generator.py
        - Monitoring dashboards
    """

    norm = _normalize_idk(answer)

    if q_type == "unknown":
        return norm not in {"i dont know", "i do not know"}

    # known → must be grounded
    return not _is_grounded_core(answer, chunks)
