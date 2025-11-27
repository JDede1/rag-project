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

    import torch
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

    if "Answer" in text:
        text = text.split("Answer")[-1]

    forbidden = [
        "ONLY using the context", "Do NOT", "Use only the context",
        "system:", "assistant:", "Context:", "Question:",
    ]
    for f in forbidden:
        text = text.replace(f, "")

    return text.strip()


# =========================================================
# -------- GROUNDING LOGIC --------------------------------
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
    if not answer:
        return False

    ans = answer.lower().strip()

    if ans in {"i don't know", "i don't know."}:
        return True

    if not chunks:
        return False

    ctx_text = " ".join(chunks).lower()

    if ans.rstrip(".") in ctx_text:
        return True

    if set(_PHONE_RE.findall(ans)) & set(_PHONE_RE.findall(ctx_text)):
        return True

    if set(_NUMBER_RE.findall(ans)) & set(_NUMBER_RE.findall(ctx_text)):
        return True

    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx_text))
    overlap = [t for t in ans_tokens if t in ctx_tokens]

    return len(overlap) >= 2


def grounding_details(answer: str, chunks: List[str]) -> Dict:
    if not answer or not chunks:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    ctx_text = " ".join(chunks).lower()
    ans_text = answer.lower().rstrip(".")

    ans_tokens = _simple_tokens(ans_text)
    ctx_tokens = set(_simple_tokens(ctx_text))

    if not ans_tokens:
        return {"grounded": False, "grounding_score": 0.0, "context_overlap": 0.0}

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    overlap_ratio = len(overlap) / max(1, len(ans_tokens))

    grounded_flag = is_grounded(answer, chunks)
    grounding_score = (0.7 * int(grounded_flag)) + (0.3 * overlap_ratio)

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
