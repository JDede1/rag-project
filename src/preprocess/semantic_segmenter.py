"""
semantic_segmenter.py
-------------------------------------------------------
Reusable segmentation utilities for RAG preprocessing.

Goal:
    Turn an answer string into a small number of coherent,
    retrieval-friendly chunks.

Design:
    1. Split answer into sentences.
    2. Group sentences into segments such that:
         - Each segment length is between min_chars and max_chars.
         - Step-like lines (1., 2., -, etc.) can start new segments
           when appropriate.
    3. Return a list of chunk strings, ready for embedding.

Intended use:
    from src.preprocess.semantic_segmenter import segment_answer_into_chunks
"""

from __future__ import annotations

import re
from typing import List


# Basic sentence splitter pattern:
# - Splits on punctuation followed by space and capital letter.
SENTENCE_REGEX = r"(?<=[.!?])\s+(?=[A-Z])"


def split_into_sentences(text: str) -> List[str]:
    """
    Split raw text into sentences using a lightweight heuristic.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    parts = re.split(SENTENCE_REGEX, text.strip())
    sentences = [p.strip() for p in parts if len(p.strip()) > 1]
    return sentences


def is_step_line(sentence: str) -> bool:
    """
    Detect whether a sentence looks like a step or list item, e.g.:

        - "1. Call us at 1-800-769-2512."
        - "- Go to Online Banking."
        - "• Select your card."

    This helps us avoid joining unrelated steps into a single chunk.
    """
    s = sentence.strip()

    if not s:
        return False

    # Bullet points or numbered items
    if re.match(r"^(\d+[\).\]]\s+|-|\*)\s*", s):
        return True

    # Simple "Step 1" style markers
    if s.lower().startswith("step "):
        return True

    return False


def group_sentences_into_segments(
    sentences: List[str],
    max_chars: int,
    min_chars: int,
) -> List[str]:
    """
    Group sentences into segments respecting the min/max character limits
    and step boundaries.

    Rules:
        - Try to keep each segment <= max_chars.
        - Avoid segments shorter than min_chars when possible.
        - If a new sentence is a "step line" and current buffer
          already has enough content, start a new segment.
    """
    if not sentences:
        return []

    segments: List[str] = []
    buffer = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # If this sentence looks like a new step and buffer is already
        # reasonably long, flush buffer to its own segment.
        if is_step_line(sent) and buffer and len(buffer) >= min_chars:
            segments.append(buffer.strip())
            buffer = sent
            continue

        # If buffer empty, start with this sentence
        if not buffer:
            buffer = sent
            continue

        # If adding this sentence stays within max_chars, append
        if len(buffer) + 1 + len(sent) <= max_chars:
            buffer += " " + sent
        else:
            # Flush and start a new buffer
            segments.append(buffer.strip())
            buffer = sent

    if buffer:
        segments.append(buffer.strip())

    # Second pass: ensure no tiny segments (merge forward where needed)
    final_segments: List[str] = []
    carry = ""

    for seg in segments:
        if len(seg) < min_chars:
            if carry:
                carry += " " + seg
            else:
                carry = seg
            continue

        if carry:
            final_segments.append(carry.strip())
            carry = ""

        final_segments.append(seg.strip())

    if carry:
        final_segments.append(carry.strip())

    return final_segments


def segment_answer_into_chunks(
    answer: str,
    max_chars: int = 320,
    min_chars: int = 80,
) -> List[str]:
    """
    High-level utility: given a raw answer string, return chunk strings.

    Parameters
    ----------
    answer : str
        The full answer text to segment.
    max_chars : int
        Maximum length of each chunk.
    min_chars : int
        Minimum desired length of each chunk.

    Returns
    -------
    List[str]
        A list of chunk strings.
    """
    sentences = split_into_sentences(answer)
    if not sentences:
        return []

    segments = group_sentences_into_segments(
        sentences=sentences,
        max_chars=max_chars,
        min_chars=min_chars,
    )
    return segments
