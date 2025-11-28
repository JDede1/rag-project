# =========================================================
# generator.py — STRICT LITERAL MODE (Option B + Style 2)
# =========================================================

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
ENFORCE_GROUNDING = os.getenv("ENFORCE_GROUNDING", "true").lower().strip() == "true"

if USE_LOCAL:
    import torch


# =========================================================
# GROQ LOADING
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
# LOCAL MODEL (lazy loading)
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
    return False


STOPWORDS = {
    "the","is","a","to","of","and","in","for","on","by","you",
    "your","or","we","with","at","from","as","an","it","be",
    "are","this","that","can",
}

def _simple_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


# =========================================================
# OPTION B — STRICT TOPIC MATCHING
# =========================================================

def _question_matches_context(question: str, chunks: List[str]) -> bool:
    if not chunks:
        return False

    q_tokens = set(_simple_tokens(question))
    if not q_tokens:
        return False

    q_lower = question.lower()

    for chunk in chunks:
        c_lower = chunk.lower()
        c_tokens = set(_simple_tokens(chunk))

        if "lost" in q_lower and ("lost" in c_lower or "stolen" in c_lower):
            return True
        if "stolen" in q_lower and ("stolen" in c_lower or "lost" in c_lower):
            return True

        if q_tokens & c_tokens:
            return True

    return False


# =========================================================
# PROMPT (STRICT LITERAL, STYLE 2 — MINIMAL TRIM)
# =========================================================

def build_prompt(question: str, chunks: List[str]) -> str:
    context_text = (
        "No context available."
        if not chunks else "\n\n".join(_attach_citations(chunks))
    )

    return (
        "You are a strict RBC banking assistant.\n"
        "RULES:\n"
        "1. Use ONLY the provided Context. No outside knowledge.\n"
        "2. Copy context sentences literally, with minimal trimming allowed.\n"
        "3. If the information is not explicitly in Context, answer: I don't know.\n"
        "4. Every factual sentence MUST have a citation [CIT:x].\n"
        "5. REQUIRED FORMAT:\n"
        "   Short Answer: <one literal or minimally trimmed sentence> [CIT:x]\n"
        "   Details:\n"
        "   • ...\n"
        "   Important Notes:\n"
        "   • ...\n"
        "   Sources:\n"
        "   • CIT:x\n"
        "Do NOT mention rules.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# =========================================================
# EXTRACT ANSWER
# =========================================================

_CIT_PATTERN = re.compile(r"\[?CIT:(\d+)\]?")

def _enforce_single_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0].strip() if parts else text.strip()


def extract_answer(full: str) -> str:
    if not full:
        return ""

    text = full.strip().replace("\r", "").replace("\n\n", "\n")

    lower = text.lower()
    sa_idx = lower.find("short answer:")

    if sa_idx != -1:
        text = text[sa_idx:]
    else:
        a_idx = lower.find("answer:")
        if a_idx != -1:
            text = text[a_idx + len("answer:"):].strip()

    cleaned = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if any(s.lower().startswith(k) for k in ["context:", "question:", "system:", "assistant:", "user:"]):
            continue
        cleaned.append(s)

    text = "\n".join(cleaned).strip()

    sections = {
        "short_answer": [],
        "details": [],
        "notes": [],
        "sources": [],
    }
    current = None

    for line in text.split("\n"):
        l = line.strip()
        ll = l.lower()

        if ll.startswith("short answer:"):
            current = "short_answer"
            content = l[len("Short Answer:"):].strip()
            if content:
                sections["short_answer"].append(content)
            continue

        if ll.startswith("details:"):
            current = "details"
            continue

        if ll.startswith("important notes:"):
            current = "notes"
            continue

        if ll.startswith("sources:"):
            current = "sources"
            continue

        if current:
            sections[current].append(l)

    # Short Answer
    sa = " ".join(sections["short_answer"]).strip()
    if not sa:
        sa = "I don't know."

    sa = _enforce_single_sentence(sa)
    short = f"Short Answer: {sa}"

    # Details
    details = []
    for l in sections["details"]:
        s = l.strip()
        if not s.startswith("•"):
            s = f"• {s}"
        details.append(s)
    if not details:
        details.append("• (no additional details)")

    # Notes
    notes = []
    for l in sections["notes"]:
        s = l.strip()
        if not s.startswith("•"):
            s = f"• {s}"
        notes.append(s)
    if not notes:
        notes.append("• (no additional notes)")

    # Sources
    body = "\n".join([short] + details + notes)
    used = sorted({int(m.group(1)) for m in _CIT_PATTERN.finditer(body)})
    if used:
        sources = [f"• CIT:{cid}" for cid in used]
    else:
        sources = ["• CIT:1"]

    final = (
        short
        + "\nDetails:\n" + "\n".join(details)
        + "\nImportant Notes:\n" + "\n".join(notes)
        + "\nSources:\n" + "\n".join(sources)
    )

    return final.strip()


# =========================================================
# GROUNDING
# =========================================================

_PHONE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
_NUM = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")

def is_grounded(answer: str, chunks: List[str]) -> bool:
    if not answer:
        return False
    ans = answer.lower().strip()

    if ans in {"i don't know", "i don't know."}:
        return True

    if not chunks:
        return False

    ctx = " ".join(chunks).lower()

    if ans.rstrip(".") in ctx:
        return True

    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx))
    if not ans_tokens:
        return False

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    ratio = len(overlap) / max(1, len(ans_tokens))

    ans_ph = set(_PHONE.findall(ans))
    ctx_ph = set(_PHONE.findall(ctx))
    ans_num = set(_NUM.findall(ans))
    ctx_num = set(_NUM.findall(ctx))

    if ans_ph and not ctx_ph:
        return False
    if ans_num and not ctx_num:
        return False

    if (ans_ph & ctx_ph or ans_num & ctx_num) and ratio >= 0.20:
        return True

    return ratio >= 0.30 and len(overlap) >= 4


def grounding_details(answer: str, chunks: List[str]) -> Dict:
    if not answer:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    ans = answer.lower().strip()
    if ans in {"i don't know", "i don't know."}:
        return {"grounded": True, "grounding_score": 1.0, "context_overlap": 0.0}

    if not chunks:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    ctx = " ".join(chunks).lower()
    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx))

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    ratio = len(overlap) / max(1, len(ans_tokens))

    grounded = is_grounded(answer, chunks)
    score = 0.7 * int(grounded) + 0.3 * ratio

    return {
        "grounded": grounded,
        "grounding_score": round(min(1.0, score), 4),
        "context_overlap": round(ratio, 4),
    }


# =========================================================
# GENERATION
# =========================================================

def generate_answer(question: str, chunks: List[str]) -> Tuple[str, Dict]:
    if not chunks or _detect_contradiction(chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, [])

    if not _question_matches_context(question, chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    prompt = build_prompt(question, chunks)

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

        new_ids = out[0][ilen:]
        raw = tok.decode(new_ids, skip_special_tokens=True)
        answer = extract_answer(raw)

    else:
        try:
            raw = _generate_groq(prompt)
            answer = extract_answer(raw)
        except Exception:
            safe = "I don't know."
            return safe, grounding_details(safe, chunks)

    details = grounding_details(answer, chunks)

    if ENFORCE_GROUNDING and not details["grounded"]:
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    return answer.strip(), details
