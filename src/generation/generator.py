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

# NEW — evaluation override
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
    text = " ".join(chunks).lower()
    if "no fee" in text and "fee" in text and "no fee" not in text.split("fee")[0]:
        return True
    return False

# =========================================================
# PROMPT
# =========================================================

def build_prompt(question: str, chunks: List[str]) -> str:
    context_text = "No context available." if not chunks else "\n\n".join(_attach_citations(chunks))

    return (
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

# =========================================================
# extract_answer() — strict but preserves structure
# =========================================================

def extract_answer(full_output: str) -> str:
    text = full_output.strip()

    lower = text.lower()
    if "answer:" in lower:
        idx = lower.index("answer:")
        text = text[idx + len("answer:"):].strip()

    forbidden_starts = (
        "context:", "question:", "system:", "assistant:",
        "you are a strict banking assistant"
    )

    cleaned = []
    for line in text.split("\n"):
        l = line.strip()
        ll = l.lower()
        if any(ll.startswith(fx) for fx in forbidden_starts):
            continue
        cleaned.append(l)

    text = "\n".join(cleaned).strip()

    while text.startswith(":"):
        text = text[1:].lstrip()

    return text.strip()

# =========================================================
# GROUNDING LOGIC (Normal Strictness)
# =========================================================

_STOP = {
    "the","is","a","to","of","and","in","for","on","by",
    "you","your","or","we","with","at","from","as","an","it","be",
}

_PHONE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
_NUM = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")

def _simple_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOP]

def is_grounded(answer: str, chunks: List[str]) -> bool:
    if not answer:
        return False

    ans = answer.lower().strip()
    if ans in {"i don't know", "i don't know."}:
        return True

    if not chunks:
        return False

    ctx = " ".join(chunks).lower()
    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx))

    if not ans_tokens:
        return False

    if set(_PHONE.findall(ans)) - set(_PHONE.findall(ctx)):
        return False

    if set(_NUM.findall(ans)) - set(_NUM.findall(ctx)):
        return False

    if ans.rstrip(".") in ctx:
        return True

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap) / len(ans_tokens)
    foreign_ratio = 1 - overlap_ratio

    return overlap_ratio >= 0.35 and foreign_ratio <= 0.5


def grounding_details(answer: str, chunks: List[str]) -> Dict:
    if not answer or not chunks:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    ctx = " ".join(chunks).lower()
    ans = answer.lower().rstrip(".")

    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx))

    if not ans_tokens:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap) / len(ans_tokens)

    foreign_ratio = 1 - overlap_ratio

    numeric_ok = (
        set(_PHONE.findall(ans)).issubset(set(_PHONE.findall(ctx))) and
        set(_NUM.findall(ans)).issubset(set(_NUM.findall(ctx)))
    )

    grounded_flag = is_grounded(answer, chunks)

    adjusted = overlap_ratio * max(0, 1 - foreign_ratio)
    if numeric_ok:
        adjusted = min(1.0, adjusted + 0.1)

    score = 0.7 * int(grounded_flag) + 0.3 * adjusted

    return {
        "grounded": grounded_flag,
        "grounding_score": round(score, 4),
        "context_overlap": round(overlap_ratio, 4),
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
    if not chunks or _detect_contradiction(chunks):
        return "I don't know.", grounding_details("I don't know.", [])

    prompt = build_prompt(question, chunks)

    if USE_LOCAL:
        tokenizer, model = _load_local_model()
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=250,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                repetition_penalty=1.05,
            )

        full_output = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = extract_answer(full_output)

    else:
        try:
            full_output = _generate_groq(prompt)
            answer = extract_answer(full_output)
        except Exception:
            return "I don't know.", grounding_details("I don't know.", chunks)

    # =========================================================
    # FINAL SAFETY GATE — now controlled by ENFORCE_GROUNDING
    # =========================================================
    details = grounding_details(answer, chunks)

    if ENFORCE_GROUNDING and not details["grounded"]:
        return "I don't know.", grounding_details("I don't know.", chunks)

    return answer.strip(), details

# =========================================================
# MANUAL TEST
# =========================================================

if __name__ == "__main__":
    test_chunks = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "We will block the card and issue a replacement."
    ]

    q = "How do I report a lost credit card?"
    ans, met = generate_answer(q, test_chunks)
    print(ans)
    print(met)
