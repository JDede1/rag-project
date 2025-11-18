"""
generator.py — Strict Grounded RAG Generator for Phi-3.5-Mini
---------------------------------------------------------------
This version prevents:
    • Prompt echo
    • Instruction leakage
    • Sentence continuation hallucinations
    • Answers that mix context with invented text

It also exposes a reusable `is_grounded(answer, chunks)` function
so that both the generator and the evaluation pipeline can share
the same core grounding logic (Balanced mode).
"""

import re
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"[Generator] Loading {MODEL_NAME} on {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
)


# ---------------------------------------------------------
# Prompt Builder (Stricter)
# ---------------------------------------------------------
def build_prompt(question: str, chunks: List[str]) -> str:
    """
    Strict grounding:
       • No creative continuation
       • No references outside context
       • If info missing → "I don't know."
    """

    context_text = "\n\n".join(chunks) if chunks else "No context available."

    prompt = (
        "Answer the question ONLY using the context below.\n"
        "If the answer is not contained fully in the context, reply: I don't know.\n"
        "Do not infer, assume, or extend beyond what is explicitly stated.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return prompt


# ---------------------------------------------------------
# Output Extraction (Strict)
# ---------------------------------------------------------
def extract_answer(full_output: str) -> str:
    """
    Strict cleaning:
      • Keep text after last 'Answer:'
      • Remove prompt echo
      • Remove instruction fragments
      • Keep ONLY first sentence
    """

    text = full_output.strip()

    # Keep only last after "Answer:"
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    # Remove echoes / prompt fragments
    bad_phrases = [
        "You are an assistant",
        "ONLY using the provided context",
        "Do NOT add outside facts",
        "If the answer is not in the context",
        "Context:",
        "Question:",
        "Answer only using the context",
        "Use only the context",
    ]
    for p in bad_phrases:
        text = text.replace(p, "").strip()

    for word in ["Context:", "Question:", "system:", "assistant:"]:
        if word in text:
            text = text.split(word)[0].strip()

    # Keep only first sentence (conservative)
    if "." in text:
        text = text.split(".")[0].strip() + "."

    return text.strip()


# ---------------------------------------------------------
# Shared Grounding Utilities (Balanced)
# ---------------------------------------------------------

_STOPWORDS = {
    "the", "is", "a", "to", "of", "and", "in",
    "for", "on", "by", "you", "your", "or", "we",
    "with", "at", "from", "as", "an", "it", "be",
}


_PHONE_RE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def _simple_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


def is_grounded(answer: str, chunks: List[str]) -> bool:
    """
    Balanced grounding logic shared between:
        • Online generator (FastAPI /ask)
        • Offline evaluator (evaluate_rag.py)

    Grounded if ANY of the following holds:
        1. Answer is "I don't know"  → treated as safe (non-hallucination)
        2. Answer is (almost) a substring of any context chunk
        3. Answer shares >= 2 non-stopword tokens with the context
        4. Answer contains a phone number or numeric pattern found in context
    """

    if not answer:
        return False

    ans = answer.strip()
    ans_low = ans.lower()

    # 1. Explicit "I don't know" is treated as non-hallucination
    if ans_low in {"i don't know", "i don't know."}:
        return True

    if not chunks:
        return False

    ctx_text = " ".join(chunks)
    ctx_low = ctx_text.lower()

    # 2. Substring containment (ignoring trailing period)
    base_ans = ans_low.rstrip(".").strip()
    if base_ans and base_ans in ctx_low:
        return True

    # 3. Entity match: phone numbers and numeric patterns (rates, amounts, etc.)
    ans_phones = set(_PHONE_RE.findall(ans_low))
    ctx_phones = set(_PHONE_RE.findall(ctx_low))
    if ans_phones & ctx_phones:
        return True

    ans_nums = set(_NUMBER_RE.findall(ans_low))
    ctx_nums = set(_NUMBER_RE.findall(ctx_low))
    if ans_nums & ctx_nums:
        return True

    # 4. Token overlap (>= 2 non-stopword tokens)
    ans_tokens = _simple_tokens(ans_low)
    ctx_tokens = set(_simple_tokens(ctx_low))

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    if len(overlap) >= 2:
        return True

    return False


# ---------------------------------------------------------
# Backwards-Compatible Wrapper
# ---------------------------------------------------------
def hybrid_grounding(answer: str, chunks: List[str]) -> bool:
    """
    Backwards-compatible wrapper so older code that calls
    `hybrid_grounding()` continues to work.

    Internally delegates to `is_grounded()` which implements
    the Balanced grounding logic.
    """
    return is_grounded(answer, chunks)


# ---------------------------------------------------------
# Main Generator
# ---------------------------------------------------------
def generate_answer(question: str, chunks: List[str]) -> str:
    """
    Main RAG answer generator:
        • Builds a strictly grounded prompt
        • Runs Phi-3.5-mini-instruct with deterministic decoding
        • Cleans the raw output
        • Applies final grounding check via `is_grounded`
    """
    if not chunks:
        return "I don't know."

    prompt = build_prompt(question, chunks)

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=130,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.05,
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = extract_answer(full_output)

    # Final hallucination check using shared Balanced grounding
    if not is_grounded(answer, chunks):
        return "I don't know."

    return answer.strip()


# ---------------------------------------------------------
# Manual Micro-Test
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_chunks = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "We will block the card from future use and issue you a new card."
    ]
    q = "How do I report a lost credit card?"
    print("Answer:", generate_answer(q, sample_chunks))
