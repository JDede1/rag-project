"""
generator.py — Final Production-Safe Grounded RAG Generator

Modes:
    • Local mode (Colab): Phi-3.5-Mini-Instruct
    • Cloud Run mode: Groq-hosted LLM (LLaMA3)

Environment Variables:
    GEN_MODE = "local" or "groq"
    GROQ_API_KEY = "<secret>" (Cloud Run only)

This version is compatible with:
    • groq==0.36.0
    • Cloud Run CPU containers
    • Streamlit frontend
    • Strict grounding, deterministic outputs
"""

import os
import re
from typing import List, Dict, Tuple

# =========================================================
# -------- ENVIRONMENT CONFIG -----------------------------
# =========================================================

GEN_MODE = os.getenv("GEN_MODE", "local").lower().strip()
USE_LOCAL = GEN_MODE == "local"      # Colab
USE_GROQ = GEN_MODE == "groq"        # Cloud Run

GROQ_MODEL = "llama3-8b-8192"


# =========================================================
# -------- OPTIONAL TORCH IMPORT (SAFE) -------------------
# =========================================================
# This avoids: "name 'torch' is not defined" during local mode.
# In Cloud Run (GEN_MODE=groq), this block never executes.

if USE_LOCAL:
    import torch


# =========================================================
# -------- TRY LOADING GROQ (CLOUD MODE) ------------------
# =========================================================

GROQ_AVAILABLE = False
GROQ_CLIENT = None

if USE_GROQ:
    try:
        from groq import Groq

        GROQ_CLIENT = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )
        GROQ_AVAILABLE = True
        print("[Generator] Groq client initialized (production mode).")

    except Exception as e:
        print("[Generator] ERROR initializing Groq client:", str(e))
        GROQ_AVAILABLE = False


# =========================================================
# -------- LAZY LOAD LOCAL PHI MODEL (COLAB) --------------
# =========================================================

_tokenizer = None
_model = None


def _load_local_model():
    """Lazy-loads Phi-3.5 model only when first needed."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    from transformers import AutoTokenizer, AutoModelForCausalLM

    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[Generator] Lazy-loading {MODEL_NAME} on {DEVICE}...")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    return _tokenizer, _model


# =========================================================
# -------- UTILITY FUNCTIONS ------------------------------
# =========================================================

def _attach_citations(chunks: List[str]) -> List[str]:
    tagged = []
    for i, chunk in enumerate(chunks):
        tagged.append(f"[CIT:{i+1}] {chunk.strip()}")
    return tagged


def _detect_contradiction(chunks: List[str]) -> bool:
    text = " ".join(chunks).lower()
    if ("no fee" in text and "fee" in text and "no fee" not in text.split("fee")[0]):
        return True
    return False


# =========================================================
# -------- PROMPT CONSTRUCTION ----------------------------
# =========================================================

def build_prompt(question: str, chunks: List[str]) -> str:
    if not chunks:
        context_text = "No context available."
    else:
        tagged = _attach_citations(chunks)
        context_text = "\n\n".join(tagged)

    prompt = (
        "You are a strict banking assistant. Follow these rules:\n"
        "1. Use ONLY the provided context. No outside facts.\n"
        "2. If the needed information is not in the context, answer EXACTLY: I don't know.\n"
        "3. Every factual claim MUST include a citation like [CIT:1].\n"
        "4. Use this structure:\n"
        "   Short Answer:\n"
        "   • 1–2 sentence summary\n"
        "   Details:\n"
        "   • Bullet points strictly from context\n"
        "   Important Notes:\n"
        "   • Clarifications from context only\n"
        "   Sources:\n"
        "   • List citation IDs used\n"
        "Do NOT mention the rules.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt


# =========================================================
# -------- OUTPUT CLEANER ---------------------------------
# =========================================================

def extract_answer(full_output: str) -> str:
    text = full_output.strip()

    # If the model echoed "Answer:" keep everything after it
    if "Answer" in text:
        text = text.split("Answer", 1)[-1]

    forbidden = [
        "ONLY using the context",
        "Do NOT",
        "Use only the context",
        "system:",
        "assistant:",
        "Context:",
        "Question:",
    ]
    for f in forbidden:
        text = text.replace(f, "")

    # Remove leftover leading colons / spaces
    text = text.strip()
    text = text.lstrip(": ").strip()

    return text.strip()


# =========================================================
# -------- GROUNDING LOGIC (NORMAL STRICTNESS) ------------
# =========================================================

_STOPWORDS = {
    "the", "is", "a", "to", "of", "and", "in", "for", "on",
    "by", "you", "your", "or", "we", "with", "at", "from",
    "as", "an", "it", "be",
}

_PHONE_RE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def _simple_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


def is_grounded(answer: str, chunks: List[str]) -> bool:
    """
    Normal strictness (Option A):

    - "I don't know" is always allowed.
    - Every phone number in the answer must appear in the context.
    - Every numeric token in the answer must appear in the context.
    - At least 35% of content tokens must overlap with the context.
    - No more than 50% of content tokens may be foreign to the context.
    """
    if not answer:
        return False

    ans = answer.lower().strip()

    # Always allow "I don't know."
    if ans in {"i don't know", "i don't know."}:
        return True

    if not chunks:
        return False

    ctx_text = " ".join(chunks).lower()

    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx_text))

    if not ans_tokens:
        return False

    # Phone and numeric safety: everything in the answer must exist in context
    ans_phones = set(_PHONE_RE.findall(ans))
    ctx_phones = set(_PHONE_RE.findall(ctx_text))
    if ans_phones and not ans_phones.issubset(ctx_phones):
        return False

    ans_nums = set(_NUMBER_RE.findall(ans))
    ctx_nums = set(_NUMBER_RE.findall(ctx_text))
    if ans_nums and not ans_nums.issubset(ctx_nums):
        return False

    # Exact substring match: safe
    base_ans = ans.rstrip(".")
    if base_ans and base_ans in ctx_text:
        return True

    # Overlap vs. foreign tokens
    overlap = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap) / max(1, len(ans_tokens))

    foreign_tokens = [t for t in ans_tokens if t not in ctx_tokens]
    foreign_ratio = len(foreign_tokens) / max(1, len(ans_tokens))

    # Normal strictness thresholds
    if overlap_ratio >= 0.35 and foreign_ratio <= 0.50:
        return True

    return False


def grounding_details(answer: str, chunks: List[str]) -> Dict:
    """
    Returns a summary used for monitoring & evaluation.
    Uses the same logic as is_grounded() but exposes
    overlap and a soft grounding_score.
    """
    if not answer or not chunks:
        return {
            "grounded": False,
            "grounding_score": 0.0,
            "context_overlap": 0.0,
        }

    ctx_text = " ".join(chunks).lower()
    ans_text = answer.lower().rstrip(".")

    ans_tokens = _simple_tokens(ans_text)
    ctx_tokens = set(_simple_tokens(ctx_text))

    if not ans_tokens:
        return {
            "grounded": False,
            "grounding_score": 0.0,
            "context_overlap": 0.0,
        }

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap) / max(1, len(ans_tokens))

    foreign_tokens = [t for t in ans_tokens if t not in ctx_tokens]
    foreign_ratio = len(foreign_tokens) / max(1, len(ans_tokens))

    # Numeric / phone consistency
    ans_phones = set(_PHONE_RE.findall(ans_text))
    ctx_phones = set(_PHONE_RE.findall(ctx_text))
    ans_nums = set(_NUMBER_RE.findall(ans_text))
    ctx_nums = set(_NUMBER_RE.findall(ctx_text))

    numeric_ok = (
        (not ans_phones or ans_phones.issubset(ctx_phones))
        and (not ans_nums or ans_nums.issubset(ctx_nums))
    )

    grounded_flag = is_grounded(answer, chunks)

    # Soft score: overlap adjusted by how many foreign tokens there are,
    # plus a small bonus if numeric_ok is true.
    adjusted_overlap = overlap_ratio * max(0.0, 1.0 - foreign_ratio)
    if numeric_ok:
        adjusted_overlap = min(1.0, adjusted_overlap + 0.1)

    grounding_score = (0.7 * int(grounded_flag)) + (0.3 * adjusted_overlap)

    return {
        "grounded": grounded_flag,
        "grounding_score": round(float(grounding_score), 4),
        "context_overlap": round(float(overlap_ratio), 4),
    }


# =========================================================
# -------- GROQ GENERATION (NEW API v0.36.0) --------------
# =========================================================

def _generate_groq(prompt: str) -> str:
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq client not available in this environment.")

    response = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=300,
    )

    return response.choices[0].message.content


# =========================================================
# -------- MAIN GENERATION FUNCTION ------------------------
# =========================================================

def generate_answer(question: str, chunks: List[str]) -> Tuple[str, Dict]:
    if not chunks or _detect_contradiction(chunks):
        return "I don't know.", grounding_details("I don't know.", [])

    prompt = build_prompt(question, chunks)

    # Local Phi (Colab)
    if USE_LOCAL:
        tokenizer, model = _load_local_model()

        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=250,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                repetition_penalty=1.05,
            )

        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = extract_answer(full_output)

    # Groq (Cloud Run)
    else:
        try:
            full_output = _generate_groq(prompt)
            answer = extract_answer(full_output)
        except Exception:
            return "I don't know.", grounding_details("I don't know.", chunks)

    # Grounding enforcement
    if not is_grounded(answer, chunks):
        return "I don't know.", grounding_details("I don't know.", chunks)

    return answer.strip(), grounding_details(answer, chunks)


# =========================================================
# -------- MANUAL TEST ------------------------------------
# =========================================================

if __name__ == "__main__":
    test_chunks = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "We will block the card and issue a replacement."
    ]
    q = "How do I report a lost credit card?"

    ans, met = generate_answer(q, test_chunks)
    print("ANSWER:\n", ans)
    print("\nMETRICS:\n", met)
