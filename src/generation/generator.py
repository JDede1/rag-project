"""
generator.py — Final Production-Safe Grounded RAG Generator
"""

import os
import re
from typing import List, Dict, Tuple

# =========================================================
# ENVIRONMENT CONFIG
# =========================================================

GEN_MODE = os.getenv("GEN_MODE", "local").lower().strip()
USE_LOCAL = GEN_MODE == "local"
USE_GROQ = GEN_MODE == "groq"

GROQ_MODEL = "llama3-8b-8192"

# Evaluation / production switch
ENFORCE_GROUNDING = os.getenv("ENFORCE_GROUNDING", "true").lower().strip() == "true"

if USE_LOCAL:
    import torch

# =========================================================
# GROQ CLIENT LOADING
# =========================================================

GROQ_AVAILABLE = False
GROQ_CLIENT = None

if USE_GROQ:
    try:
        from groq import Groq
        GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
        GROQ_AVAILABLE = True
    except Exception:
        GROQ_AVAILABLE = False

# =========================================================
# LOCAL PHI MODEL (Lazy Load)
# =========================================================

_tokenizer = None
_model = None


def _load_local_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL = "microsoft/Phi-3.5-mini-instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[Generator] Loading {MODEL} on {DEVICE}...")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    return _tokenizer, _model

# =========================================================
# UTILITIES
# =========================================================


def _attach_citations(chunks: List[str]) -> List[str]:
    return [f"[CIT:{i+1}] {chunk.strip()}" for i, chunk in enumerate(chunks)]


def _detect_contradiction(chunks: List[str]) -> bool:
    """
    Very conservative contradiction detector.
    Currently kept minimal to avoid false positives.
    """
    text = " ".join(chunks).lower()
    # You can extend this later if you add robust contradiction patterns.
    return False

# =========================================================
# PROMPT
# =========================================================


def build_prompt(question: str, chunks: List[str]) -> str:
    """
    Build a prompt that:
      - Uses ONLY provided context
      - Enforces strict answer format
      - Avoids leaking rules / template bullets into the answer
    """
    context_text = "No context available." if not chunks else "\n\n".join(
        _attach_citations(chunks)
    )

    return (
        "You are a strict banking assistant for RBC FAQs.\n"
        "Follow these instructions:\n"
        "1. Use ONLY the provided context below. Do NOT use any outside knowledge.\n"
        "2. If the needed information is not in the context, answer exactly: I don't know.\n"
        "3. Every factual statement that comes from the context MUST include an inline "
        "citation like [CIT:1] next to the sentence it supports.\n"
        "4. Do NOT repeat or list the context chunks in your answer. Just answer the question.\n"
        "5. Use the following headings exactly, in this order:\n"
        "   Short Answer:\n"
        "   Details:\n"
        "   Important Notes:\n"
        "   Sources:\n"
        "6. Under Details, Important Notes, and Sources, use bullet points that start with '• '.\n"
        "7. Do NOT mention these instructions in your answer.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer using ONLY the context, in the exact format specified above."
    )

# =========================================================
# extract_answer() — strict but preserves structure
# =========================================================


def extract_answer(full_output: str) -> str:
    """
    Preserve the model's structure:

        Short Answer:
        Details:
        Important Notes:
        Sources:

    Remove:
      - Any lead-up text before 'Short Answer:'
      - Prompt echoes such as 'Context:', 'Question:', 'system:', 'assistant:'
      - Stray 'Answer:' prefixes
    """
    if not full_output:
        return ""

    text = full_output.strip().replace("\r\n", "\n").replace("\r", "\n")
    lower = text.lower()

    # Prefer to start at the first occurrence of 'Short Answer:'
    sa_idx = lower.find("short answer:")
    if sa_idx != -1:
        text = text[sa_idx:]
    else:
        # Fallback: strip anything before 'Answer:'
        ans_idx = lower.find("answer:")
        if ans_idx != -1:
            text = text[ans_idx + len("answer:") :].lstrip()

    # Remove prompt echoes / meta-instructions line by line
    forbidden_starts = (
        "context:",
        "question:",
        "system:",
        "assistant:",
        "user:",
        "you are a strict banking assistant",
        "you are a helpful assistant",
    )

    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(stripped)
            continue

        ll = stripped.lower()
        if any(ll.startswith(prefix) for prefix in forbidden_starts):
            continue

        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines).strip()

    # Remove any stray leading colons
    while text.startswith(":"):
        text = text[1:].lstrip()

    return text.strip()

# =========================================================
# GROUNDING LOGIC (Safer but less brittle)
# =========================================================

_STOP = {
    "the",
    "is",
    "a",
    "to",
    "of",
    "and",
    "in",
    "for",
    "on",
    "by",
    "you",
    "your",
    "or",
    "we",
    "with",
    "at",
    "from",
    "as",
    "an",
    "it",
    "be",
}

# Accept common North American phone patterns (3-3-4) with optional separators.
_PHONE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
_NUM = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def _simple_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOP]


def is_grounded(answer: str, chunks: List[str]) -> bool:
    """
    Determines if a model answer is reasonably grounded in the retrieved context.

    Rules:
      • "I don't know." is always considered safe.
      • If there is no context, any non-"I don't know." answer is ungrounded.
      • Direct substring match ⇒ grounded.
      • Numeric and phone overlaps are treated as strong evidence.
      • Requires a modest token overlap (>= 3 tokens and >= 0.25 ratio).
      • If the answer introduces phones or numbers and the context has none, it's ungrounded.
    """
    if not answer:
        return False

    ans = answer.lower().strip()
    if ans in {"i don't know", "i don't know."}:
        return True

    if not chunks:
        return False

    ctx_text = " ".join(chunks).lower()

    # Direct containment (e.g., short factual snippet fully from context)
    base_ans = ans.rstrip(".")
    if base_ans and base_ans in ctx_text:
        return True

    # Token overlap
    ans_tokens = _simple_tokens(ans)
    if not ans_tokens:
        return False

    ctx_tokens = set(_simple_tokens(ctx_text))
    overlap_tokens = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap_tokens) / max(1, len(ans_tokens))

    # Phone and numeric evidence
    ans_phones = set(_PHONE.findall(ans))
    ctx_phones = set(_PHONE.findall(ctx_text))
    phone_overlap = bool(ans_phones and ctx_phones and (ans_phones & ctx_phones))

    ans_nums = set(_NUM.findall(ans))
    ctx_nums = set(_NUM.findall(ctx_text))
    num_overlap = bool(ans_nums and ctx_nums and (ans_nums & ctx_nums))

    # If the answer uses phones/numbers but context has none → ungrounded
    if ans_phones and not ctx_phones:
        return False
    if ans_nums and not ctx_nums:
        return False

    # Strong numeric/phone overlap ⇒ grounded
    if phone_overlap or num_overlap:
        return True

    # Otherwise require reasonable lexical overlap
    return overlap_ratio >= 0.25 and len(overlap_tokens) >= 3


def grounding_details(answer: str, chunks: List[str]) -> Dict:
    """
    Returns:
      - grounded (bool)
      - grounding_score (0–1)
      - context_overlap (0–1, lexical)
    """
    if not answer:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    ans = answer.lower().strip()
    # Treat "I don't know." as a safe, fully grounded fallback, even with no chunks.
    if ans in {"i don't know", "i don't know."}:
        return {"grounded": True, "grounding_score": 1.0, "context_overlap": 0.0}

    if not chunks:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    ctx_text = " ".join(chunks).lower()

    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx_text))

    if not ans_tokens:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    overlap_tokens = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap_tokens) / max(1, len(ans_tokens))

    # Numeric/phone evidence
    ans_phones = set(_PHONE.findall(ans))
    ctx_phones = set(_PHONE.findall(ctx_text))
    phone_overlap = bool(ans_phones and ctx_phones and (ans_phones & ctx_phones))

    ans_nums = set(_NUM.findall(ans))
    ctx_nums = set(_NUM.findall(ctx_text))
    num_overlap = bool(ans_nums and ctx_nums and (ans_nums & ctx_nums))

    grounded_flag = is_grounded(answer, chunks)

    # Base score: heavy weight on boolean grounded flag, some on lexical overlap
    base_score = 0.7 * int(grounded_flag) + 0.3 * overlap_ratio

    # Slight boost if we have strong numeric/phone evidence
    if phone_overlap or num_overlap:
        base_score = min(1.0, base_score + 0.1)

    return {
        "grounded": grounded_flag,
        "grounding_score": round(float(base_score), 4),
        "context_overlap": round(float(overlap_ratio), 4),
    }

# =========================================================
# GROQ GENERATION
# =========================================================


def _generate_groq(prompt: str) -> str:
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq client unavailable.")

    resp = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=300,
    )

    return resp.choices[0].message.content

# =========================================================
# MAIN GENERATION FUNCTION
# =========================================================


def generate_answer(question: str, chunks: List[str]) -> Tuple[str, Dict]:
    """
    Main entry point used by FastAPI and evaluation.

    Returns:
      answer (str) in the strict format:
          Short Answer:
          Details:
          Important Notes:
          Sources:

      metadata (dict) from grounding_details()
    """
    # If we have no useful context or a detected contradiction, fall back safely.
    if not chunks or _detect_contradiction(chunks):
        safe_answer = "I don't know."
        return safe_answer, grounding_details(safe_answer, [])

    prompt = build_prompt(question, chunks)

    # Local Phi (Colab / dev)
    if USE_LOCAL:
        tokenizer, model = _load_local_model()
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Decode ONLY the generated tokens (exclude the prompt),
        # so the answer does not contain template + context text.
        input_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=250,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                repetition_penalty=1.05,
            )

        generated_ids = out[0][input_len:]
        full_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        answer = extract_answer(full_output)

    # Groq (Cloud Run)
    else:
        try:
            full_output = _generate_groq(prompt)
            answer = extract_answer(full_output)
        except Exception:
            safe_answer = "I don't know."
            return safe_answer, grounding_details(safe_answer, chunks)

    details = grounding_details(answer, chunks)

    # Enforce grounding in production mode
    if ENFORCE_GROUNDING and not details["grounded"]:
        safe_answer = "I don't know."
        return safe_answer, grounding_details(safe_answer, chunks)

    return answer.strip(), details

# =========================================================
# MANUAL TEST
# =========================================================

if __name__ == "__main__":
    test_chunks = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "We will block the card and issue a replacement.",
    ]

    q = "How do I report a lost credit card?"
    ans, met = generate_answer(q, test_chunks)
    print("=== ANSWER ===")
    print(ans)
    print("\n=== METRICS ===")
    print(met)
