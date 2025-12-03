# =========================================================
# generator.py — Strict Literal Mode (No LLM)
# =========================================================

import re
from typing import List, Dict, Tuple

# ---------------------------------------------------------
# Token utilities
# ---------------------------------------------------------
STOPWORDS = {
    "the", "is", "a", "to", "of", "and", "in", "for", "on", "by", "you",
    "your", "or", "we", "with", "at", "from", "as", "an", "it", "be", "are",
    "this", "that", "can", "if", "would", "will"
}

def _simple_tokens(text: str) -> List[str]:
    return [
        t for t in re.findall(r"\w+", text.lower())
        if t not in STOPWORDS
    ]


# ---------------------------------------------------------
# Build citations
# ---------------------------------------------------------
def _attach_citations(chunks: List[str]) -> List[str]:
    return [f"[CIT:{i+1}] {c.strip()}" for i, c in enumerate(chunks)]


# ---------------------------------------------------------
# Sentence helper
# ---------------------------------------------------------
def _first_sentence(text: str) -> str:
    """
    Extract only FIRST literal sentence from a chunk.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0].strip() if parts else text.strip()


# ---------------------------------------------------------
# MAIN STRICT LITERAL ANSWER BUILDER
# ---------------------------------------------------------
def generate_answer(question: str, chunks: List[str]) -> Tuple[str, Dict]:
    """
    Strict literal RAG generator:
        • No LLM calls
        • No paraphrasing
        • No creativity
        • Only literal RBC text
    """

    # No context at all
    if not chunks:
        safe = "I don't know."
        return safe, {
            "grounded": True,
            "grounding_score": 1.0,
            "context_overlap": 0.0,
        }

    # Attach citations
    cited_chunks = _attach_citations(chunks)

    # Pick literal first sentence from best chunk
    short_sentence = _first_sentence(chunks[0])
    short_answer = f"Short Answer: {short_sentence} [CIT:1]"

    # Build details: each chunk becomes a literal bullet
    details = []
    for i, c in enumerate(chunks, start=1):
        literal = c.strip()
        details.append(f"• [CIT:{i}] {literal}")

    notes = ["• (no additional information)"]

    sources = [f"• CIT:{i}" for i in range(1, len(chunks) + 1)]

    final_answer = (
        short_answer + "\n"
        "Details:\n" + "\n".join(details) + "\n"
        "Important Notes:\n" + "\n".join(notes) + "\n"
        "Sources:\n" + "\n".join(sources)
    ).strip()

    # Grounding: always perfectly grounded because all text is literal context
    details_info = {
        "grounded": True,
        "grounding_score": 1.0,
        "context_overlap": 1.0,
    }

    return final_answer, details_info
