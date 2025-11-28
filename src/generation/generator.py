# =========================================================
# generator.py — RAG Generator (Strict Mode)
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
# UTILS
# =========================================================

def _attach_citations(chunks: List[str]) -> List[str]:
    return [f"[CIT:{i+1}] {chunk.strip()}" for i, chunk in enumerate(chunks)]

def _detect_contradiction(chunks: List[str]) -> bool:
    return False


# =========================================================
# STRICT TOKENIZATION (shared)
# =========================================================

STOPWORDS = {
    "the","is","a","to","of","and","in","for","on","by","you","your","or",
    "we","with","at","from","as","an","it","be","are","this","that","can",
}

def _simple_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


# =========================================================
# OPTION B — IMPROVED STRICT TOPIC MATCHING
# =========================================================
def _question_matches_context(question: str, chunks: List[str]) -> bool:
    """
    Option B strict literal mode:
    - Per-chunk lexical overlap instead of entire-context union
    - Minimal token overlap required: >= 1 with ANY chunk
    - Also allow direct substring hints (e.g., "lost card")
    """

    if not chunks:
        return False

    q_tokens = set(_simple_tokens(question))
    if not q_tokens:
        return False

    question_lower = question.lower()

    for chunk in chunks:
        chunk_lower = chunk.lower()
        c_tokens = set(_simple_tokens(chunk))

        # 1. Direct phrase hints
        if "lost" in question_lower and ("lost" in chunk_lower or "stolen" in chunk_lower):
            return True

        if "stolen" in question_lower and ("stolen" in chunk_lower or "lost" in chunk_lower):
            return True

        # 2. Any token overlap per chunk
        if q_tokens & c_tokens:
            return True

    return False


# =========================================================
# PROMPT CONSTRUCTION (strict literal)
# =========================================================

def build_prompt(question: str, chunks: List[str]) -> str:
    context_text = (
        "No context available."
        if not chunks else "\n\n".join(_attach_citations(chunks))
    )

    return (
        "You are a strict banking assistant for RBC FAQs.\n"
        "RULES:\n"
        "1. Use ONLY the provided Context. No outside knowledge.\n"
        "2. Do NOT infer or paraphrase beyond literal context.\n"
        "3. If information is not present, answer exactly: I don't know.\n"
        "4. Every factual sentence MUST include a citation [CIT:x].\n"
        "5. Structure:\n"
        "   Short Answer: <sentence> [CIT:x]\n"
        "   Details:\n"
        "   • ...\n"
        "   Important Notes:\n"
        "   • ...\n"
        "   Sources:\n"
        "   • CIT:x\n"
        "Do NOT mention instructions.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

# =========================================================
# extract_answer()
# =========================================================

_CIT_PATTERN = re.compile(r"\[?CIT:(\d+)\]?")

def _enforce_single_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[0].strip() if sentences else text.strip()

def extract_answer(full_output: str) -> str:
    if not full_output:
        return ""

    text = full_output.strip().replace("\r\n", "\n")
    lower = text.lower()

    sa_idx = lower.find("short answer:")
    if sa_idx != -1:
        text = text[sa_idx:]
    else:
        ans_idx = lower.find("answer:")
        if ans_idx != -1:
            text = text[ans_idx + len("answer:") :].strip()

    forbidden = (
        "context:", "question:", "system:", "assistant:", "user:",
        "you are a strict banking assistant",
    )
    cleaned = []
    for line in text.split("\n"):
        s = line.strip()
        if any(s.lower().startswith(f) for f in forbidden):
            continue
        cleaned.append(s)

    text = "\n".join(cleaned).strip()

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
            content = l[len("Short Answer:"):].strip()
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

    sa_text = " ".join(sections["short_answer"]).strip()
    if not sa_text:
        sa_text = "I don't know."

    sa_text = _enforce_single_sentence(sa_text)
    short_answer_line = f"Short Answer: {sa_text}"

    details_lines = [
        f"• {l.strip()}" if not l.strip().startswith("•") else l.strip()
        for l in sections["details"] if l.strip()
    ]
    if not details_lines:
        details_lines.append("• (no additional details)")

    notes_lines = [
        f"• {l.strip()}" if not l.strip().startswith("•") else l.strip()
        for l in sections["important_notes"] if l.strip()
    ]
    if not notes_lines:
        notes_lines.append("• (no additional notes)")

    used = sorted({int(m.group(1)) for m in _CIT_PATTERN.finditer(short_answer_line)})
    sources_lines = [f"• CIT:{cid}" for cid in used] if used else ["• CIT:1"]

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
# GROUNDING (unchanged)
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
    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx))

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap) / max(1, len(ans_tokens))

    ans_phones = set(_PHONE.findall(ans))
    ctx_phones = set(_PHONE.findall(ctx))

    ans_nums = set(_NUM.findall(ans))
    ctx_nums = set(_NUM.findall(ctx))

    if ans_phones and not ctx_phones:
        return False
    if ans_nums and not ctx_nums:
        return False

    if (ans_phones & ctx_phones or ans_nums & ctx_nums) and overlap_ratio >= 0.20:
        return True

    return overlap_ratio >= 0.30 and len(overlap) >= 4


def grounding_details(answer: str, chunks: List[str]) -> Dict:
    ans = answer.lower().strip()
    if ans in {"i don't know", "i don't know."}:
        return {"grounded": True, "grounding_score": 1.0, "context_overlap": 0.0}

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
# GENERATE ANSWER
# =========================================================

def generate_answer(question: str, chunks: List[str]) -> Tuple[str, Dict]:

    if not chunks or _detect_contradiction(chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, [])

    # OPTION B — improved strict matching
    if not _question_matches_context(question, chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    prompt = build_prompt(question, chunks)

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

        full_output = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        answer = extract_answer(full_output)

    else:
        try:
            full_output = _generate_groq(prompt)
            answer = extract_answer(full_output)
        except Exception:
            safe = "I don't know."
            return safe, grounding_details(safe, chunks)

    details = grounding_details(answer, chunks)

    if ENFORCE_GROUNDING and not details["grounded"]:
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    return answer.strip(), details
