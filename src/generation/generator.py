"""
generator.py — RAG Generator
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
    return False  # intentionally minimal


# =========================================================
# STRICT TOPIC MATCHING (OPTION A — BEST PRACTICE)
# =========================================================

STOPWORDS = {
    "the","is","a","to","of","and","in","for","on","by","you","your","or",
    "we","with","at","from","as","an","it","be","are","this","that","can",
}

def _simple_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def _question_matches_context(question: str, chunks: List[str]) -> bool:
    """
    STRICT MODE (Option A):
      - If question and context have ZERO meaningful token overlap → reject
      - No semantic inference. Literal lexical check only.
    """
    if not chunks:
        return False

    q_tokens = set(_simple_tokens(question))
    ctx_tokens = set(_simple_tokens(" ".join(chunks)))

    if not q_tokens or not ctx_tokens:
        return False

    overlap = q_tokens & ctx_tokens

    return len(overlap) >= 1  # strict mode: require >=1 exact token match


# =========================================================
# PROMPT — STRICT LITERAL MODE
# =========================================================

def build_prompt(question: str, chunks: List[str]) -> str:
    context_text = "No context available." if not chunks else "\n\n".join(
        _attach_citations(chunks)
    )

    return (
        "You are a strict banking assistant for RBC FAQs.\n"
        "RULES:\n"
        "1. Use ONLY the provided Context. No outside knowledge.\n"
        "2. Do NOT paraphrase, summarize creatively, or infer missing steps.\n"
        "   - Only use wording that appears in the Context.\n"
        "3. If information is not directly present, answer exactly: I don't know.\n"
        "4. Every factual sentence MUST include its citation like [CIT:1].\n"
        "5. Use EXACTLY this structure:\n"
        "   Short Answer: <one sentence from Context> [CIT:x]\n"
        "   Details:\n"
        "   • ...\n"
        "   Important Notes:\n"
        "   • ...\n"
        "   Sources:\n"
        "   • CIT:x\n"
        "Do NOT copy or repeat context chunks verbatim as paragraphs.\n"
        "Do NOT mention these instructions.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

# =========================================================
# extract_answer()
# =========================================================

_CIT_PATTERN = re.compile(r"\[?CIT:(\d+)\]?")

def _enforce_single_sentence(text: str) -> str:
    """Return ONLY the first complete sentence."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return text.strip()
    return sentences[0].strip()


def extract_answer(full_output: str) -> str:
    """Normalize RAG output to required structure."""
    if not full_output:
        return ""

    text = full_output.strip().replace("\r\n", "\n")

    # Start at Short Answer
    lower = text.lower()
    sa_idx = lower.find("short answer:")
    if sa_idx != -1:
        text = text[sa_idx:]
    else:
        ans_idx = lower.find("answer:")
        if ans_idx != -1:
            text = text[ans_idx + len("answer:") :].strip()

    # Remove prompt echoes
    forbidden = (
        "context:", "question:", "system:", "assistant:", "user:",
        "you are a strict banking assistant",
    )
    cleaned = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if any(s.lower().startswith(f) for f in forbidden):
            continue
        cleaned.append(s)

    text = "\n".join(cleaned).strip()

    # ---- Parse sections ----
    sections = {
        "short_answer": [],
        "details": [],
        "important_notes": [],
        "sources": [],
    }
    current = None

    for line in text.split("\n"):
        l = line.strip()
        ll = l.lower()

        if ll.startswith("short answer:"):
            current = "short_answer"
            content = l[len("Short Answer:") :].strip()
            if content:
                sections["short_answer"].append(content)
            continue

        if ll.startswith("details:"):
            current = "details"
            continue

        if ll.startswith("important notes:"):
            current = "important_notes"
            continue

        if ll.startswith("sources:"):
            current = "sources"
            continue

        if current:
            sections[current].append(l)

    # -------------------------
    # Short Answer normalization
    # -------------------------
    sa_text = " ".join(sections["short_answer"]).strip()
    if sa_text.startswith(("•", "-", "*")):
        sa_text = sa_text[1:].strip()

    if not sa_text:
        # fallback to first detail
        if sections["details"]:
            d = sections["details"][0]
            if d.startswith("•"):
                d = d[1:].strip()
            sa_text = d.strip()
        else:
            sa_text = "I don't know."

    # ENFORCE ONLY ONE SENTENCE
    sa_text = _enforce_single_sentence(sa_text)

    short_answer_line = f"Short Answer: {sa_text}"

    # -------------------------
    # Details
    # -------------------------
    details_lines = []
    for l in sections["details"]:
        s = l.strip()
        if not s:
            continue
        if not s.startswith("•"):
            s = f"• {s}"
        details_lines.append(s)

    if not details_lines:
        details_lines.append("• (no additional details)")

    # -------------------------
    # Important Notes
    # -------------------------
    notes_lines = []
    for l in sections["important_notes"]:
        s = l.strip()
        if not s:
            continue
        if not s.startswith("•"):
            s = f"• {s}"
        notes_lines.append(s)

    if not notes_lines:
        notes_lines.append("• (no additional notes)")

    # -------------------------
    # Sources
    # -------------------------
    output_body = "\n".join([short_answer_line] + details_lines + notes_lines)
    used = sorted({int(m.group(1)) for m in _CIT_PATTERN.finditer(output_body)})

    sources_lines = [f"• CIT:{cid}" for cid in used] if used else ["• CIT:1"]

    # -------------------------
    # Final assembly
    # -------------------------
    final = (
        short_answer_line
        + "\nDetails:\n"
        + "\n".join(details_lines)
        + "\nImportant Notes:\n"
        + "\n".join(notes_lines)
        + "\nSources:\n"
        + "\n".join(sources_lines)
    )

    return final.strip()

# =========================================================
# GROUNDING LOGIC
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

    # Direct literal containment
    if ans.rstrip(".") in ctx:
        return True

    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx))

    if not ans_tokens:
        return False

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap) / max(1, len(ans_tokens))

    ans_phones = set(_PHONE.findall(ans))
    ctx_phones = set(_PHONE.findall(ctx))

    ans_nums = set(_NUM.findall(ans))
    ctx_nums = set(_NUM.findall(ctx))

    # Reject if answer invents numbers/phones
    if ans_phones and not ctx_phones:
        return False
    if ans_nums and not ctx_nums:
        return False

    # Numeric/phone support + minimal lexical overlap
    if (ans_phones & ctx_phones or ans_nums & ctx_nums) and overlap_ratio >= 0.20:
        return True

    # General threshold
    return overlap_ratio >= 0.30 and len(overlap) >= 4


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
    overlap_ratio = len(overlap) / max(1, len(ans_tokens))

    grounded_flag = is_grounded(answer, chunks)
    base = 0.7 * int(grounded_flag) + 0.3 * overlap_ratio

    return {
        "grounded": grounded_flag,
        "grounding_score": round(min(1.0, base), 4),
        "context_overlap": round(overlap_ratio, 4),
    }

# =========================================================
# GROQ GENERATION
# =========================================================

def _generate_groq(prompt: str) -> str:
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq unavailable")

    resp = GROQ_CLIENT.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=350,
    )

    return resp.choices[0].message.content

# =========================================================
# MAIN GENERATION FUNCTION
# =========================================================

def generate_answer(question: str, chunks: List[str]) -> Tuple[str, Dict]:
    """
    MAIN ENTRY POINT — used by FastAPI + evaluation
    """
    # 1. No context or contradiction → safe fallback
    if not chunks or _detect_contradiction(chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, [])

    # 2. STRICT TOPIC CONSISTENCY (Option A)
    if not _question_matches_context(question, chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    # 3. Build strict prompt
    prompt = build_prompt(question, chunks)

    # 4. Generate
    if USE_LOCAL:
        tokenizer, model = _load_local_model()
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
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

        new_ids = out[0][input_len:]
        full_output = tokenizer.decode(new_ids, skip_special_tokens=True)
        answer = extract_answer(full_output)

    else:
        try:
            full_output = _generate_groq(prompt)
            answer = extract_answer(full_output)
        except Exception:
            safe = "I don't know."
            return safe, grounding_details(safe, chunks)

    # 5. Grounding enforcement
    details = grounding_details(answer, chunks)
    if ENFORCE_GROUNDING and not details["grounded"]:
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    return answer.strip(), details


# Manual Test
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
