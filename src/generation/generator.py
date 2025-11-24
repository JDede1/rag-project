"""
generator.py — Grounded RAG Generator for Phi-3.5-Mini

This module:
    • Builds strict grounded prompts
    • Generates structured answers with citations
    • Detects contradictions and enforces safe fallback
    • Computes grounding metrics (grounded flag, score, context token overlap)
    • Returns a tuple: (answer: str, grounding_info: dict)
"""

import re
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------
# Model Setup
# ---------------------------------------------------------

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"[Generator] Loading {MODEL_NAME} on {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
)


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def _attach_citations(chunks: List[str]) -> List[str]:
    """Attach deterministic citation IDs to each text chunk."""
    tagged = []
    for i, chunk in enumerate(chunks):
        cid = f"[CIT:{i+1}]"
        tagged.append(f"{cid} {chunk.strip()}")
    return tagged


def _detect_contradiction(chunks: List[str]) -> bool:
    """Simple heuristic for contradictions inside retrieved chunks."""
    text = " ".join(chunks).lower()
    if ("no fee" in text and "fee" in text and "no fee" not in text.split("fee")[0]):
        return True
    return False


# ---------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------

def build_prompt(question: str, chunks: List[str]) -> str:
    """Construct a strict, grounded prompt with citation-tagged context."""
    if not chunks:
        context_text = "No context available."
    else:
        tagged = _attach_citations(chunks)
        context_text = "\n\n".join(tagged)

    prompt = (
        "You are a strict banking assistant. Follow the rules:\n"
        "1. Use ONLY the provided context. No outside facts.\n"
        "2. If the needed information is not in the context, answer EXACTLY: I don't know.\n"
        "3. Every factual claim MUST include its citation like [CIT:1].\n"
        "4. Structure the response exactly as:\n"
        "   Short Answer:\n"
        "   • 1–2 sentence summary\n"
        "   Details:\n"
        "   • Bullet points strictly from context\n"
        "   Important Notes:\n"
        "   • Clarifications (only if found in context)\n"
        "   Sources:\n"
        "   • List citation IDs used\n"
        "Do not mention the rules or the prompt.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return prompt


# ---------------------------------------------------------
# Output Extraction
# ---------------------------------------------------------

def extract_answer(full_output: str) -> str:
    """Extract the model’s answer while removing unwanted system echoes."""
    text = full_output.strip()

    if "Answer" in text:
        text = text.split("Answer")[-1]

    forbidden = [
        "ONLY using the context",
        "Do NOT add",
        "Use only the context",
        "You are",
        "system:",
        "assistant:",
        "Context:",
        "Question:",
    ]

    for f in forbidden:
        text = text.replace(f, "").strip()

    return text.strip()


# ---------------------------------------------------------
# Grounding Logic
# ---------------------------------------------------------

_STOPWORDS = {
    "the", "is", "a", "to", "of", "and", "in",
    "for", "on", "by", "you", "your", "or", "we",
    "with", "at", "from", "as", "an", "it", "be",
}

_PHONE_RE = re.compile(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")


def _simple_tokens(text: str) -> List[str]:
    """Tokenize while removing stopwords."""
    if not text:
        return []
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


def is_grounded(answer: str, chunks: List[str]) -> bool:
    """Checks whether the answer content can be verified inside retrieved chunks."""
    if not answer:
        return False

    ans = answer.lower().strip()

    if ans in {"i don't know", "i don't know."}:
        return True

    if not chunks:
        return False

    ctx_text = " ".join(chunks).lower()

    # Direct substring
    base_ans = ans.rstrip(".").strip()
    if base_ans and base_ans in ctx_text:
        return True

    # Phone numbers
    if set(_PHONE_RE.findall(ans)) & set(_PHONE_RE.findall(ctx_text)):
        return True

    # Numbers
    if set(_NUMBER_RE.findall(ans)) & set(_NUMBER_RE.findall(ctx_text)):
        return True

    # Token overlap
    ans_tokens = _simple_tokens(ans)
    ctx_tokens = set(_simple_tokens(ctx_text))

    overlap = [t for t in ans_tokens if t in ctx_tokens]
    return len(overlap) >= 2


def hybrid_grounding(answer: str, chunks: List[str]) -> bool:
    """Alias for compatibility."""
    return is_grounded(answer, chunks)


# ---------------------------------------------------------
# Grounding Metrics
# ---------------------------------------------------------

def grounding_details(answer: str, chunks: List[str]) -> Dict:
    """
    Compute grounding metrics:
        • grounded (bool)
        • grounding_score (0–1)
        • context_overlap (0–1)
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

    grounded_flag = is_grounded(answer, chunks)

    # Weighted grounding score
    grounding_score = (0.7 * int(grounded_flag)) + (0.3 * overlap_ratio)

    return {
        "grounded": grounded_flag,
        "grounding_score": round(float(grounding_score), 4),
        "context_overlap": round(float(overlap_ratio), 4),
    }


# ---------------------------------------------------------
# Main Generation Function
# ---------------------------------------------------------

def generate_answer(question: str, chunks: List[str]) -> Tuple[str, Dict]:
    """
    Generate an answer using Phi-3.5-Mini, then compute grounding metrics.

    Returns:
        answer (str)
        grounding_info (dict)
    """
    # No context or contradiction → fallback
    if not chunks:
        return "I don't know.", grounding_details("I don't know.", [])

    if _detect_contradiction(chunks):
        return "I don't know.", grounding_details("I don't know.", chunks)

    # Build prompt
    prompt = build_prompt(question, chunks)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate deterministic text
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=250,
            temperature=None,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.05,
        )

    # Decode and extract clean answer
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = extract_answer(full_output)

    # Final hallucination guard
    if not is_grounded(answer, chunks):
        return "I don't know.", grounding_details("I don't know.", chunks)

    return answer.strip(), grounding_details(answer, chunks)


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------

if __name__ == "__main__":
    test_chunks = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "We will block the card from future use and issue you a new card."
    ]
    q = "How do I report a lost credit card?"

    answer, metrics = generate_answer(q, test_chunks)
    print("Answer:\n", answer)
    print("Metrics:\n", metrics)
