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
    # Extend this later if you add robust contradiction patterns.
    return False

# =========================================================
# PROMPT — STRICT LITERAL MODE
# =========================================================


def build_prompt(question: str, chunks: List[str]) -> str:
    """
    Strict literal mode:
      - Use ONLY sentences or phrases that appear in the context.
      - Do NOT paraphrase, infer, or invent any new information.
      - Short Answer must be a single sentence derived directly from context.
    """
    context_text = "No context available." if not chunks else "\n\n".join(
        _attach_citations(chunks)
    )

    return (
        "You are a strict banking assistant for RBC FAQs.\n"
        "Your behavior MUST follow these rules exactly:\n"
        "1. Use ONLY the provided Context below. Do NOT use any outside knowledge.\n"
        "2. Do NOT paraphrase, summarize creatively, or infer missing steps.\n"
        "   - Every sentence in your answer MUST be directly supported by the Context.\n"
        "   - Prefer copying or minimally trimming context sentences.\n"
        "3. If the needed information is not present in the Context, answer exactly: I don't know.\n"
        "4. Every factual statement that comes from the Context MUST include an inline citation\n"
        "   in the form [CIT:1] next to the sentence it supports.\n"
        "5. Do NOT repeat or list the Context chunks themselves. Just answer the question.\n"
        "6. Use the following headings exactly, in this order:\n"
        "   Short Answer:\n"
        "   Details:\n"
        "   Important Notes:\n"
        "   Sources:\n"
        "7. Short Answer MUST be a single sentence on the same line as the heading, like:\n"
        "   Short Answer: <one sentence taken from or strictly supported by the Context> [CIT:x]\n"
        "   - Do NOT use bullet points in the Short Answer.\n"
        "8. Under Details and Important Notes, use bullet points that start with '• '.\n"
        "9. Under Sources, list each citation as a bullet in the form:\n"
        "   Sources:\n"
        "   • CIT:1\n"
        "   • CIT:2\n"
        "10. Do NOT mention or describe these instructions in your answer.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer using ONLY the Context, in the exact format specified above."
    )

# =========================================================
# extract_answer() — normalize structure & sources
# =========================================================

# Pattern to detect citations like [CIT:1], CIT:1, [CIT:12], etc.
_CIT_PATTERN = re.compile(r"\[?CIT:(\d+)\]?")


def extract_answer(full_output: str) -> str:
    """
    Preserve and normalize the model's structure:

        Short Answer:
        Details:
        Important Notes:
        Sources:

    Steps:
      - Remove any lead-up text before 'Short Answer:' or 'Answer:'.
      - Strip prompt echoes such as 'Context:', 'Question:', 'system:', 'assistant:'.
      - Normalize:
          • Short Answer → one-line sentence, no bullets.
          • Details / Important Notes → bullet list with '• ' prefix.
          • Sources → bullet list of '• CIT:x' for citations actually used.
    """
    if not full_output:
        return ""

    # Normalize line breaks
    text = full_output.strip().replace("\r\n", "\n").replace("\r", "\n")
    lower = text.lower()

    # Prefer to start at 'Short Answer:'
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

    cleaned_lines: List[str] = []
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

    # ---- Parse into sections ------------------------------------------------
    lines = text.split("\n")
    sections = {
        "short_answer": [],
        "details": [],
        "important_notes": [],
        "sources": [],
    }

    current = None

    def _starts_with(line: str, header: str) -> bool:
        return line.lower().startswith(header)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            # Preserve blank lines within sections, but not before any heading.
            if current:
                sections[current].append("")
            continue

        ll = line.lower()
        if _starts_with(ll, "short answer:"):
            current = "short_answer"
            content = line[len("Short Answer:") :].strip()
            if content:
                sections["short_answer"].append(content)
            continue
        elif _starts_with(ll, "details:"):
            current = "details"
            content = line[len("Details:") :].strip()
            if content:
                sections["details"].append(content)
            continue
        elif _starts_with(ll, "important notes:"):
            current = "important_notes"
            content = line[len("Important Notes:") :].strip()
            if content:
                sections["important_notes"].append(content)
            continue
        elif _starts_with(ll, "sources:"):
            current = "sources"
            content = line[len("Sources:") :].strip()
            if content:
                sections["sources"].append(content)
            continue

        if current:
            sections[current].append(line)

    # ---- Normalize Short Answer ----------------------------------------------
    sa_parts = [l for l in sections["short_answer"] if l.strip()]
    short_answer_text = " ".join(sa_parts).strip()

    # Remove any bullet markers in Short Answer
    if short_answer_text.startswith(("•", "-", "*")):
        short_answer_text = short_answer_text[1:].lstrip()

    # If still empty, try to salvage from Details first bullet
    if not short_answer_text and sections["details"]:
        first_detail = sections["details"][0].strip()
        if first_detail.startswith(("•", "-", "*")):
            first_detail = first_detail[1:].lstrip()
        short_answer_text = first_detail

    # Final fallback if still empty
    if not short_answer_text:
        short_answer_text = "I don't know."

    short_answer_line = f"Short Answer: {short_answer_text}"

    # ---- Normalize Details ---------------------------------------------------
    details_lines: List[str] = []
    for l in sections["details"]:
        content = l.strip()
        if not content:
            continue
        if not content.startswith("•"):
            content = f"• {content}"
        details_lines.append(content)

    # ---- Normalize Important Notes ------------------------------------------
    notes_lines: List[str] = []
    for l in sections["important_notes"]:
        content = l.strip()
        if not content:
            continue
        if not content.startswith("•"):
            content = f"• {content}"
        notes_lines.append(content)

    # ---- Collect citations used outside Sources -----------------------------
    body_text = "\n".join(
        [short_answer_line]
        + details_lines
        + notes_lines
    )

    used_cits = sorted(
        {int(m.group(1)) for m in _CIT_PATTERN.finditer(body_text)}
    )

    # ---- Normalize Sources ---------------------------------------------------
    sources_lines: List[str] = []

    if used_cits:
        for cid in used_cits:
            sources_lines.append(f"• CIT:{cid}")
    else:
        # If no citations were detected, keep any existing non-empty sources lines,
        # but normalize to bullet form. (Grounding enforcement may still reject
        # the answer later if it is not properly grounded.)
        for l in sections["sources"]:
            content = l.strip()
            if not content:
                continue
            if not content.startswith("•"):
                content = f"• {content}"
            sources_lines.append(content)

    # Ensure we always have at least one Sources line
    if not sources_lines:
        sources_lines.append("• CIT:1")

    # ---- Reassemble final answer --------------------------------------------
    out_lines: List[str] = [short_answer_line, "Details:"]
    if details_lines:
        out_lines.extend(details_lines)
    else:
        out_lines.append("• (no additional details)")

    out_lines.append("Important Notes:")
    if notes_lines:
        out_lines.extend(notes_lines)
    else:
        out_lines.append("• (no additional notes)")

    out_lines.append("Sources:")
    out_lines.extend(sources_lines)

    final_text = "\n".join(out_lines).strip()
    return final_text

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

    Strict literal mode rules:
      • "I don't know." is always considered safe.
      • If there is no context, any non-"I don't know." answer is ungrounded.
      • Direct substring match ⇒ grounded.
      • Numeric and phone overlaps are strong evidence but NOT sufficient alone.
      • Requires a stronger token overlap (>= 4 tokens and >= 0.30 ratio).
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

    # Numeric/phone overlap is strong evidence, but we still require some lexical overlap
    if (phone_overlap or num_overlap) and overlap_ratio >= 0.20 and len(overlap_tokens) >= 3:
        return True

    # Otherwise require stronger lexical overlap
    return overlap_ratio >= 0.30 and len(overlap_tokens) >= 4


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
