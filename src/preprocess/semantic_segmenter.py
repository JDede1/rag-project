"""
semantic_segmenter.py
-------------------------------------------------------
Segmentation utilities for RAG preprocessing.

Purpose:
    Convert an answer string into coherent, retrieval-friendly chunks.

Design:
    1. Split answer into sentences using punctuation and paragraph cues.
    2. Group sentences into segments:
         - Chunk size between min_chars and max_chars.
         - Step-like lines (e.g., "1.", "-", "*", "Step 1") trigger new segments.
    3. Return final chunk list ready for embedding.

This module expects upstream steps to preserve newline structure.
"""

from __future__ import annotations

import re
from typing import List


# -------------------------------------------------------
# PARAGRAPH AND SENTENCE SPLITTING
# -------------------------------------------------------

def collapse_spaces(text: str) -> str:
    """
    Normalize spaces inside lines but preserve newline structure.
    """
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Robust sentence splitter:
        - Uses newline boundaries as soft separators.
        - Splits inside paragraphs on punctuation + capital letter.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    text = collapse_spaces(text)

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"

    sentences: List[str] = []

    for para in paragraphs:
        parts = re.split(pattern, para)
        for p in parts:
            p = p.strip()
            if len(p) > 1:
                sentences.append(p)

    return sentences


# -------------------------------------------------------
# STEP / LIST ITEM DETECTION
# -------------------------------------------------------

def is_step_line(sentence: str) -> bool:
    """
    Detect step-like or list-like sentences, which often indicate
    procedural boundaries in banking FAQs.
    """
    s = sentence.strip()
    if not s:
        return False

    # Numbered or bullet items
    if re.match(r"^(\d+[\).\]]\s+|-|\*)\s*", s):
        return True

    # "Step 1", "Step 2", etc.
    if s.lower().startswith("step "):
        return True

    # Bullet characters replaced during cleaning
    if s.startswith("- "):
        return True

    return False


# -------------------------------------------------------
# SEGMENT GROUPING
# -------------------------------------------------------

def group_sentences_into_segments(
    sentences: List[str],
    max_chars: int,
    min_chars: int,
) -> List[str]:
    """
    Group sentences into coherent segments based on character limits
    and structural cues.

    Rules:
        - Keep each segment <= max_chars when possible.
        - Avoid segments < min_chars unless unavoidable.
        - Start new segments at step-like lines if the current buffer
          already has enough content.
    """
    if not sentences:
        return []

    segments: List[str] = []
    buffer = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Start new segment at a step line if buffer is established
        if is_step_line(sent) and buffer and len(buffer) >= min_chars:
            segments.append(buffer.strip())
            buffer = sent
            continue

        # Start buffer
        if not buffer:
            buffer = sent
            continue

        # Append if within limit
        if len(buffer) + 1 + len(sent) <= max_chars:
            buffer += " " + sent
        else:
            segments.append(buffer.strip())
            buffer = sent

    if buffer:
        segments.append(buffer.strip())

    # Second pass: merge tiny segments
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


# -------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------

def segment_answer_into_chunks(
    answer: str,
    max_chars: int = 320,
    min_chars: int = 80,
) -> List[str]:
    """
    Segment an answer into retrieval-ready chunks.

    Parameters
    ----------
    answer : str
        The full answer text.
    max_chars : int
        Maximum characters per chunk.
    min_chars : int
        Minimum characters per chunk.

    Returns
    -------
    List[str]
        Chunked segments ready for embedding.
    """
    if not answer or not isinstance(answer, str):
        return []

    sentences = split_into_sentences(answer)
    if not sentences:
        return []

    chunks = group_sentences_into_segments(
        sentences=sentences,
        max_chars=max_chars,
        min_chars=min_chars,
    )

    return chunks
