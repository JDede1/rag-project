# =========================================================
# generator.py — RAG Generator (Strict Literal Mode)
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
    """
    Lazy-load Phi-3.5-mini-instruct only once per process.
    """
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
    """
    Turn a list of plain chunks into [CIT:1] ..., [CIT:2] ... format.
    """
    return [f"[CIT:{i+1}] {chunk.strip()}" for i, chunk in enumerate(chunks)]


def _detect_contradiction(chunks: List[str]) -> bool:
    """
    Very conservative placeholder – currently disabled.
    """
    return False


# =========================================================
# STRICT TOKENIZATION (shared)
# =========================================================

STOPWORDS = {
    "the", "is", "a", "to", "of", "and", "in", "for", "on", "by",
    "you", "your", "or", "we", "with", "at", "from", "as", "an",
    "it", "be", "are", "this", "that", "can",
}


def _simple_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


# =========================================================
# OPTION B — IMPROVED STRICT TOPIC MATCHING
# =========================================================

def _question_matches_context(question: str, chunks: List[str]) -> bool:
    """
    Option B strict literal mode:

    - Per-chunk lexical overlap instead of whole-context union.
    - Minimal requirement: at least 1 overlapping content token
      between the question and ANY individual chunk.
    - Extra hard-coded hints for high-risk domains like
      lost/stolen card vs. fraud.

    If this check fails, we *immediately* return "I don't know."
    even if retrieval returned something.
    """
    if not chunks:
        return False

    q_tokens = set(_simple_tokens(question))
    if not q_tokens:
        return False

    q_lower = question.lower()

    for chunk in chunks:
        c_lower = chunk.lower()
        c_tokens = set(_simple_tokens(chunk))

        # Domain-specific hints:
        # Lost/stolen card questions should match chunks about
        # "lost" / "stolen" cards, not just generic fraud.
        if "lost" in q_lower and ("lost" in c_lower or "stolen" in c_lower):
            return True
        if "stolen" in q_lower and ("stolen" in c_lower or "lost" in c_lower):
            return True

        # Generic lexical overlap with this chunk
        if q_tokens & c_tokens:
            return True

    return False


# =========================================================
# PROMPT CONSTRUCTION (strict literal)
# =========================================================

def build_prompt(question: str, chunks: List[str]) -> str:
    """
    Build the strict literal-mode prompt.
    The model must:
        - Use ONLY context
        - Not infer or paraphrase
        - Include [CIT:x] for every factual sentence
        - Follow the exact 4-section structure
    """
    context_text = (
        "No context available."
        if not chunks
        else "\n\n".join(_attach_citations(chunks))
    )

    return (
        "You are a strict banking assistant for RBC FAQs.\n"
        "RULES (you MUST follow these exactly):\n"
        "1. Use ONLY the provided Context. Do NOT use any outside knowledge.\n"
        "2. Do NOT infer, generalize, or paraphrase beyond the literal wording\n"
        "   in the Context. Prefer copying or minimally trimming context sentences.\n"
        "3. If the requested information is not explicitly present in the Context,\n"
        "   answer exactly: I don't know.\n"
        "4. Every factual sentence that comes from the Context MUST include an\n"
        "   inline citation in the form [CIT:x] right after the sentence.\n"
        "5. Your answer MUST have EXACTLY the following structure:\n"
        "   Short Answer: <one sentence taken from or strictly supported by Context> [CIT:x]\n"
        "   Details:\n"
        "   • ...\n"
        "   Important Notes:\n"
        "   • ...\n"
        "   Sources:\n"
        "   • CIT:x\n"
        "6. Do NOT mention or describe these rules in your answer.\n"
        "7. Do NOT paste or repeat the raw Context verbatim as large blocks.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# =========================================================
# extract_answer() — normalize structure
# =========================================================

_CIT_PATTERN = re.compile(r"\[?CIT:(\d+)\]?")

def _enforce_single_sentence(text: str) -> str:
    """
    Restrict Short Answer to the first complete sentence.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[0].strip() if sentences else text.strip()


def extract_answer(full_output: str) -> str:
    """
    Normalize the raw model output into the canonical structure:

        Short Answer: <sentence>
        Details:
        • ...
        Important Notes:
        • ...
        Sources:
        • CIT:x

    All bullet sections are normalized to '• '.
    Sources are derived from the citations actually used.
    """
    if not full_output:
        return ""

    # Normalize newlines
    text = full_output.strip().replace("\r\n", "\n").replace("\r", "\n")
    lower = text.lower()

    # Start at 'Short Answer:' if present
    sa_idx = lower.find("short answer:")
    if sa_idx != -1:
        text = text[sa_idx:]
    else:
        # Otherwise start after 'Answer:' if present
        ans_idx = lower.find("answer:")
        if ans_idx != -1:
            text = text[ans_idx + len("answer:"):].lstrip()

    # Strip prompt echoes
    forbidden_starts = (
        "context:",
        "question:",
        "system:",
        "assistant:",
        "user:",
        "you are a strict banking assistant",
    )
    cleaned_lines: List[str] = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            cleaned_lines.append("")
            continue
        ll = s.lower()
        if any(ll.startswith(pref) for pref in forbidden_starts):
            continue
        cleaned_lines.append(s)

    text = "\n".join(cleaned_lines).strip()

    # Parse sections
    sections = {
        "short_answer": [],
        "details": [],
        "important_notes": [],
        "sources": [],
    }
    current = None

    for line in text.split("\n"):
        l = line.strip()
        if not l:
            continue
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

    # ---- Short Answer ----
    sa_text = " ".join(sections["short_answer"]).strip()
    if not sa_text and sections["details"]:
        # Fallback: use first detail line if short answer missing
        d = sections["details"][0].strip()
        if d.startswith(("•", "-", "*")):
            d = d[1:].strip()
        sa_text = d

    if not sa_text:
        sa_text = "I don't know."

    sa_text = _enforce_single_sentence(sa_text)
    short_answer_line = f"Short Answer: {sa_text}"

    # ---- Details ----
    details_lines: List[str] = []
    for l in sections["details"]:
        s = l.strip()
        if not s:
            continue
        if not s.startswith("•"):
            s = f"• {s}"
        details_lines.append(s)
    if not details_lines:
        details_lines.append("• (no additional details)")

    # ---- Important Notes ----
    notes_lines: List[str] = []
    for l in sections["important_notes"]:
        s = l.strip()
        if not s:
            continue
        if not s.startswith("•"):
            s = f"• {s}"
        notes_lines.append(s)
    if not notes_lines:
        notes_lines.append("• (no additional notes)")

    # ---- Sources (citations used in body) ----
    body_text = "\n".join([short_answer_line] + details_lines + notes_lines)
    used = sorted({int(m.group(1)) for m in _CIT_PATTERN.finditer(body_text)})
    if used:
        sources_lines = [f"• CIT:{cid}" for cid in used]
    else:
        # Fallback to a dummy citation if model forgot
        sources_lines = ["• CIT:1"]

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
# GROUNDING LOGIC (shared with metrics / evaluation)
# =========================================================

_PHONE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
_NUM = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def is_grounded(answer: str, chunks: List[str]) -> bool:
    """
    Return True if the answer is reasonably grounded in the provided chunks.

    Rules:
      - "I don't know." is always considered safe and grounded.
      - If there are no chunks and the answer is not IDK → ungrounded.
      - Direct literal containment in context → grounded.
      - Otherwise requires sufficient lexical overlap PLUS consistency
        of phone numbers / numeric values.
    """
    if not answer:
        return False

    ans = answer.lower().strip()
    if ans in {"i don't know", "i don't know."}:
        return True

    if not chunks:
        return False

    ctx = " ".join(chunks).lower()

    # Direct literal containment (extra safe-path)
    if ans.rstrip(".") and ans.rstrip(".") in ctx:
        return True

    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx))
    if not ans_tokens:
        return False

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap) / max(1, len(ans_tokens))

    # Phone / numeric consistency
    ans_phones = set(_PHONE.findall(ans))
    ctx_phones = set(_PHONE.findall(ctx))

    ans_nums = set(_NUM.findall(ans))
    ctx_nums = set(_NUM.findall(ctx))

    # Reject if answer invents numbers/phones not in context
    if ans_phones and not ctx_phones:
        return False
    if ans_nums and not ctx_nums:
        return False

    # Strong evidence: shared phones or numbers + some lexical overlap
    if (ans_phones & ctx_phones or ans_nums & ctx_nums) and overlap_ratio >= 0.20:
        return True

    # General lexical threshold
    return overlap_ratio >= 0.30 and len(overlap) >= 4


def grounding_details(answer: str, chunks: List[str]) -> Dict:
    """
    Compute:
        grounded (bool)
        grounding_score (0–1)
        context_overlap (0–1)
    """
    if not answer:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    ans = answer.lower().strip()
    if ans in {"i don't know", "i don't know."}:
        # IDK is treated as safe, even if chunks are empty.
        return {"grounded": True, "grounding_score": 1.0, "context_overlap": 0.0}

    if not chunks:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    ctx = " ".join(chunks).lower()
    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx))

    if not ans_tokens:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

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
        raise RuntimeError("Groq client unavailable or not configured.")

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
    Main entry point used by:
        - FastAPI backend (/ask)
        - Phase 5 evaluation
        - Metrics & monitoring (via grounding_details)

    Returns:
        answer (str) in the canonical 4-section format
        metadata (dict) from grounding_details()
    """
    # 1. No context or contradiction → safe fallback
    if not chunks or _detect_contradiction(chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, [])

    # 2. Strict topic consistency (Option B)
    if not _question_matches_context(question, chunks):
        safe = "I don't know."
        return safe, grounding_details(safe, chunks)

    # 3. Build strict literal-mode prompt
    prompt = build_prompt(question, chunks)

    # 4. Generate with local Phi or Groq
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


# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":
    test_chunks = [
        "If your RBC credit card is lost or stolen, call our 24-hour toll-free number 1-800-769-2512. "
        "We’ll block the card from future use and issue you a new card.",
        "If there is a transaction on your statement that you know you didn’t make, lock your card via "
        "the RBC Mobile app or RBC Online Banking and dispute the transaction online.",
    ]
    q = "How do I report a lost credit card?"
    ans, met = generate_answer(q, test_chunks)
    print("=== ANSWER ===")
    print(ans)
    print("\n=== METRICS ===")
    print(met)
