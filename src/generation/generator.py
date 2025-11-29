# =========================================================
# generator.py — STRICT BUT USABLE LITERAL MODE (Option A)
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
# Local Phi-3.5 model (lazy load)
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
    "your","or","we","with","at","from","as","an","it","be",
    "are","this","that","can","if","would","will",
}

def _simple_tokens(text: str) -> List[str]:
    return [
        t for t in re.findall(r"\w+", text.lower())
        if t not in STOPWORDS and not t.startswith("cit")
    ]


# ---------------------------------------------------------
# Robust Topic Matching (Fixed for Lost/Stolen)
# ---------------------------------------------------------
def _question_matches_context(question: str, chunks: List[str]) -> bool:
    if not chunks:
        return False

    q = question.lower()
    q_tokens = set(_simple_tokens(question))

    for chunk in chunks:
        c = chunk.lower()
        c_tokens = set(_simple_tokens(chunk))

        # Strong signals
        if "lost" in q and ("lost" in c or "stolen" in c):
            return True
        if "stolen" in q and ("stolen" in c or "lost" in c):
            return True

        # Moderate signal: card + lost/stolen patterns
        if ("card" in q and ("lost" in c or "stolen" in c)):
            return True

        # Lexical overlap fallback
        if q_tokens & c_tokens:
            return True

    return False


# ---------------------------------------------------------
# Prompt Builder (Literal Mode)
# ---------------------------------------------------------
def _attach_citations(chunks: List[str]) -> List[str]:
    return [f"[CIT:{i+1}] {chunk.strip()}" for i, chunk in enumerate(chunks)]

def build_prompt(question: str, chunks: List[str]) -> str:
    context = "No context available." if not chunks else "\n\n".join(
        _attach_citations(chunks)
    )

    return (
        "You are a strict RBC banking assistant.\n"
        "RULES:\n"
        "1. Use ONLY the provided Context. No outside knowledge.\n"
        "2. Copy sentences LITERALLY from the context with minimal trimming.\n"
        "3. If the information is not explicitly in Context, answer ONLY: I don't know.\n"
        "4. Every factual sentence MUST have a citation [CIT:x].\n"
        "5. REQUIRED FORMAT:\n"
        "   Short Answer: <one literal sentence> [CIT:x]\n"
        "   Details:\n"
        "   • ...\n"
        "   Important Notes:\n"
        "   • ...\n"
        "   Sources:\n"
        "   • CIT:x\n"
        "Do NOT mention rules.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# ---------------------------------------------------------
# Extract Answer
# ---------------------------------------------------------
_CIT_PATTERN = re.compile(r"CIT:(\d+)", re.IGNORECASE)

def _enforce_single_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0].strip()

def extract_answer(raw: str) -> str:
    if not raw:
        return "I don't know."

    t = raw.strip()
    lower = t.lower()

    # Try to locate Short Answer
    idx = lower.find("short answer:")
    if idx != -1:
        t = t[idx:]

    # Clean lines
    lines = []
    for ln in t.split("\n"):
        s = ln.strip()
        if s and not s.lower().startswith(("context:", "question:", "system:", "assistant:", "user:")):
            lines.append(s)

    text = "\n".join(lines)

    # Parse sections
    sections = {"short": [], "details": [], "notes": [], "sources": []}
    current = None

    for ln in text.split("\n"):
        l = ln.strip()
        ll = l.lower()

        if ll.startswith("short answer:"):
            current = "short"
            content = l[len("Short Answer:"):].strip()
            if content:
                sections["short"].append(content)
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

    # Build Short Answer
    sa = " ".join(sections["short"]).strip()
    if not sa:
        sa = "I don't know."

    sa = _enforce_single_sentence(sa)
    short = f"Short Answer: {sa}"

    def bulletize(lines):
        return [l if l.startswith("•") else f"• {l}" for l in lines] or ["• (no additional information)"]

    details = bulletize(sections["details"])
    notes = bulletize(sections["notes"])

    body = "\n".join([short] + details + notes)
    used_cits = sorted({int(m.group(1)) for m in _CIT_PATTERN.finditer(body)})

    sources = [f"• CIT:{cid}" for cid in used_cits] or ["• CIT:1"]

    final = (
        short
        + "\nDetails:\n" + "\n".join(details)
        + "\nImportant Notes:\n" + "\n".join(notes)
        + "\nSources:\n" + "\n".join(sources)
    )

    return final.strip()


# ---------------------------------------------------------
# Grounding (Relaxed but Safe)
# ---------------------------------------------------------
def is_grounded(answer: str, chunks: List[str]) -> bool:
    if not answer or not chunks:
        return False

    if answer.lower().strip() == "i don't know.":
        return True

    ans_tokens = set(_simple_tokens(answer))
    ctx_tokens = set(_simple_tokens(" ".join(chunks)))

    if not ans_tokens:
        return False

    overlap = ans_tokens & ctx_tokens
    ratio = len(overlap) / max(1, len(ans_tokens))

    # More reasonable thresholds
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
# GROQ Generation
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
# MAIN: generate_answer()
# ---------------------------------------------------------
def generate_answer(question: str, chunks: List[str]) -> Tuple[str, Dict]:
    if not chunks:
        safe = "I don't know."
        return safe, grounding_details(safe, [])

    # Robust topic-match
    if not _question_matches_context(question, chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    # Build prompt
    prompt = build_prompt(question, chunks)

    # Generate answer
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

    else:
        try:
            raw = _generate_groq(prompt)
            answer = extract_answer(raw)
        except Exception:
            safe = "I don't know."
            return safe, grounding_details(safe, chunks)

    # Validate grounding
    details = grounding_details(answer, chunks)
    if ENFORCE_GROUNDING and not details["grounded"]:
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    return answer.strip(), details
