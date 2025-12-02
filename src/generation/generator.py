# =========================================================
# generator.py 
# =========================================================

import os
import re
from typing import List, Dict, Tuple

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
GEN_MODE = os.getenv("GEN_MODE", "local").lower().strip()
USE_LOCAL = GEN_MODE == "local"
USE_GROQ = GEN_MODE == "groq"

ENFORCE_GROUNDING = os.getenv("ENFORCE_GROUNDING", "true").lower().strip() == "true"
GROQ_MODEL = "llama3-8b-8192"

if USE_LOCAL:
    import torch


# ---------------------------------------------------------
# GROQ client
# ---------------------------------------------------------
GROQ_AVAILABLE = False
if USE_GROQ:
    try:
        from groq import Groq
        GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
        GROQ_AVAILABLE = True
    except Exception:
        GROQ_AVAILABLE = False


# ---------------------------------------------------------
# Local Phi-3.5 loader
# ---------------------------------------------------------
_tokenizer = None
_model = None

def _load_local_model():
    global _tokenizer, _model
    if _tokenizer is not None:
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


# ---------------------------------------------------------
# Token utilities
# ---------------------------------------------------------
STOPWORDS = {
    "the","is","a","to","of","and","in","for","on","by","you",
    "your","or","we","with","at","from","as","an","it","be","are",
    "this","that","can","if","would","will"
}

def _simple_tokens(text: str) -> List[str]:
    return [
        t for t in re.findall(r"\w+", text.lower())
        if t not in STOPWORDS and not t.startswith("cit")
    ]


# ---------------------------------------------------------
# HARDENED TOPIC MATCHING (FINAL)
# ---------------------------------------------------------
def _question_matches_context(question: str, chunks: List[str]) -> bool:

    if not chunks:
        return False

    q = question.lower()

    # LOST / STOLEN
    if "lost" in q or "stolen" in q:
        for ch in chunks:
            c = ch.lower()
            if (
                "lost" in c or "stolen" in c
                or "permanently lost" in c
                or "misplaced" in c
            ):
                return True

    # FRAUD
    if "fraud" in q or "unauthorized" in q or "dispute" in q:
        for ch in chunks:
            c = ch.lower()
            if "fraud" in c or "unauthorized" in c or "dispute" in c:
                return True

    # E-TRANSFER
    if ("transfer" in q or "e-transfer" in q or "etransfer" in q or "interac" in q):
        for ch in chunks:
            c = ch.lower()
            if "transfer" in c or "interac" in c:
                return True

    # PASSWORD / LOGIN
    if ("password" in q or "login" in q or "reset" in q):
        for ch in chunks:
            c = ch.lower()
            if "password" in c or "reset" in c or "login" in c:
                return True

    # LEXICAL fallback
    q_tokens = set(_simple_tokens(question))
    for ch in chunks:
        if q_tokens & set(_simple_tokens(ch)):
            return True

    return False


# ---------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------
def _attach_citations(chunks: List[str]) -> List[str]:
    return [f"[CIT:{i+1}] {c.strip()}" for i, c in enumerate(chunks)]

def build_prompt(question: str, chunks: List[str]) -> str:
    ctx = "No context available." if not chunks else "\n\n".join(_attach_citations(chunks))

    return (
        "You are a strict RBC banking assistant.\n"
        "RULES:\n"
        "1. Use ONLY the provided Context. No outside knowledge.\n"
        "2. Copy literal sentences ONLY.\n"
        "3. If missing info → reply exactly: I don't know.\n"
        "4. Every factual sentence MUST include a citation.\n"
        "FORMAT:\n"
        "   Short Answer: <literal sentence> [CIT:x]\n"
        "   Details:\n"
        "   • literal sentences\n"
        "   Important Notes:\n"
        "   • literal sentences\n"
        "   Sources:\n"
        "   • CIT:x\n"
        "Do NOT mention rules.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# ---------------------------------------------------------
# Extract answer
# ---------------------------------------------------------
_CIT_PATTERN = re.compile(r"CIT:(\d+)", re.IGNORECASE)

def _enforce_single_sentence(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0].strip()

def extract_answer(raw: str) -> str:
    if not raw:
        return "I don't know."

    t = raw.strip()
    lower = t.lower()

    idx = lower.find("short answer:")
    if idx != -1:
        t = t[idx:]

    cleaned = []
    for ln in t.split("\n"):
        s = ln.strip()
        if s and not s.lower().startswith(
            ("context:", "question:", "system:", "assistant:", "user:")
        ):
            cleaned.append(s)

    text = "\n".join(cleaned)

    sections = {"short": [], "details": [], "notes": []}
    current = None

    for ln in text.split("\n"):
        line = ln.strip()
        low = line.lower()

        if low.startswith("short answer:"):
            current = "short"
            v = line[len("Short Answer:") :].strip()
            if v:
                sections["short"].append(v)
            continue

        if low.startswith("details:"):
            current = "details"
            continue

        if low.startswith("important notes:"):
            current = "notes"
            continue

        if current:
            sections[current].append(line)

    sa = " ".join(sections["short"]).strip() or "I don't know."
    sa = _enforce_single_sentence(sa)
    short = f"Short Answer: {sa}"

    def literal_only(lines):
        out = []
        for l in lines:
            if "[" in l and "]" in l:
                if not l.startswith("•"):
                    l = "• " + l
                out.append(l)
        return out or ["• (no additional information)"]

    details = literal_only(sections["details"])
    notes = literal_only(sections["notes"])

    body = "\n".join([short] + details + notes)
    used = sorted({int(m.group(1)) for m in _CIT_PATTERN.finditer(body)})
    sources = [f"• CIT:{c}" for c in used] or ["• CIT:1"]

    return (
        short
        + "\nDetails:\n" + "\n".join(details)
        + "\nImportant Notes:\n" + "\n".join(notes)
        + "\nSources:\n" + "\n".join(sources)
    ).strip()


# ---------------------------------------------------------
# Option A — STRICT LITERAL FALLBACK
# ---------------------------------------------------------
def _option_a_fallback(chunks: List[str]) -> str:
    """
    Uses the FIRST chunk as the literal Short Answer.
    This guarantees the exact RBC wording with no hallucination.
    """

    if not chunks:
        return "I don't know."

    first = chunks[0].strip()
    return (
        f"Short Answer: {_enforce_single_sentence(first)} [CIT:1]\n"
        f"Details:\n• [CIT:1] {first}\n"
        f"Important Notes:\n• (no additional information)\n"
        f"Sources:\n• CIT:1"
    )


# ---------------------------------------------------------
# Grounding logic
# ---------------------------------------------------------
def is_grounded(answer: str, chunks: List[str]) -> bool:
    if answer.lower().strip() == "i don't know.":
        return True
    if not chunks:
        return False

    ans_tokens = set(_simple_tokens(answer))
    ctx_tokens = set(_simple_tokens(" ".join(chunks)))
    overlap = ans_tokens & ctx_tokens
    ratio = len(overlap) / max(1, len(ans_tokens))
    return ratio >= 0.10 and len(overlap) >= 1


def grounding_details(answer: str, chunks: List[str]) -> Dict:
    if not chunks:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    if answer.lower().strip() == "i don't know.":
        return {"grounded": True, "grounding_score": 1.0, "context_overlap": 0.0}

    ans_tokens = set(_simple_tokens(answer))
    ctx_tokens = set(_simple_tokens(" ".join(chunks)))
    overlap = ans_tokens & ctx_tokens
    ratio = len(overlap) / max(1, len(ans_tokens))

    grounded = is_grounded(answer, chunks)
    score = 0.7 * int(grounded) + 0.3 * ratio

    return {
        "grounded": grounded,
        "grounding_score": round(score, 4),
        "context_overlap": round(ratio, 4),
    }


# ---------------------------------------------------------
# GROQ generation (FIXED)
# ---------------------------------------------------------
def _generate_groq(prompt: str) -> str:
    if not GROQ_AVAILABLE:
        return "I don't know."

    out = GROQ_CLIENT.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.0,
        top_p=1.0,
        max_tokens=250,
    )
    return out.choices[0].message["content"]


# ---------------------------------------------------------
# MAIN: generate_answer() WITH FALLBACK
# ---------------------------------------------------------
def generate_answer(question: str, chunks: List[str]) -> Tuple[str, Dict]:

    if not chunks:
        safe = "I don't know."
        return safe, grounding_details(safe, [])

    if not _question_matches_context(question, chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    prompt = build_prompt(question, chunks)

    # ---------------------------
    # TRY NORMAL GENERATION
    # ---------------------------
    try:
        if USE_LOCAL:
            tok, model = _load_local_model()
            enc = tok(prompt, return_tensors="pt").to(model.device)
            ilen = enc["input_ids"].shape[1]

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=250,
                    temperature=0.0,
                    top_p=1.0,
                    do_sample=False,
                )

            raw = tok.decode(out[0][ilen:], skip_special_tokens=True)
            answer = extract_answer(raw)

        else:  # GROQ MODE
            raw = _generate_groq(prompt)
            answer = extract_answer(raw)

    except Exception:
        # HARD FALLBACK TO STRICT LITERAL
        fallback = _option_a_fallback(chunks)
        return fallback, grounding_details(fallback, chunks)


    # ---------------------------
    # VALIDATE ANSWER
    # ---------------------------
    details = grounding_details(answer, chunks)

    # If generator gave "I don't know." → fallback to literal RBC sentence
    if answer.strip().lower() == "i don't know.":
        fallback = _option_a_fallback(chunks)
        return fallback, grounding_details(fallback, chunks)

    # If ungrounded and grounding enforcement is ON → fallback
    if ENFORCE_GROUNDING and not details["grounded"]:
        fallback = _option_a_fallback(chunks)
        return fallback, grounding_details(fallback, chunks)

    return answer.strip(), details
