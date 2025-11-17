"""
generator.py — Strict Grounded RAG Generator for Phi-3.5-Mini
---------------------------------------------------------------
This version prevents:
    • Prompt echo
    • Instruction leakage
    • Sentence continuation hallucinations
    • Answers that mix context with invented text
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------
# Model Configuration
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
# Prompt Builder (Stricter)
# ---------------------------------------------------------
def build_prompt(question: str, chunks: list[str]) -> str:
    """
    Strict grounding:
       • No creative continuation
       • No references outside context
       • If info missing → "I don't know."
    """

    context_text = "\n\n".join(chunks) if chunks else "No context available."

    prompt = (
        "Answer the question ONLY using the context below.\n"
        "If the answer is not contained fully in the context, reply: I don't know.\n"
        "Do not infer, assume, or extend beyond what is explicitly stated.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return prompt


# ---------------------------------------------------------
# Output Extraction (Strict)
# ---------------------------------------------------------
def extract_answer(full_output: str) -> str:
    """
    Strict cleaning:
      • Keep text after last 'Answer:'
      • Remove prompt echo
      • Remove instruction fragments
      • Keep ONLY first sentence unless fully grounded
    """

    text = full_output.strip()

    # Keep only last after "Answer:"
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    # Remove echoes
    bad_phrases = [
        "You are an assistant",
        "ONLY using the provided context",
        "Do NOT add outside facts",
        "If the answer is not in the context",
        "Context:",
        "Question:",
        "Answer only using the context",
        "Use only the context",
    ]
    for p in bad_phrases:
        text = text.replace(p, "").strip()

    # Remove trailing prompt fragments
    for word in ["Context:", "Question:", "system:", "assistant:"]:
        if word in text:
            text = text.split(word)[0].strip()

    # Keep only first sentence
    if "." in text:
        text = text.split(".")[0].strip() + "."

    return text.strip()


# ---------------------------------------------------------
# Strict Hybrid Grounding
# ---------------------------------------------------------
def hybrid_grounding(answer: str, chunks: list[str]) -> bool:
    """
    Stricter grounding rules:
        1. Must share >= 3 non-stopword tokens with context
        2. OR answer is exact substring of any chunk
        3. OR matches a phone number from context
    """

    if not answer or answer.lower() == "i don't know":
        return True

    ans_tokens = set(answer.lower().split())
    ctx_text = " ".join(chunks).lower()
    ctx_tokens = set(ctx_text.split())

    stopwords = {
        "the", "is", "a", "to", "of", "and", "in",
        "for", "on", "by", "you", "your", "or", "we"
    }

    ans_tokens = ans_tokens - stopwords

    # Rule 1: token overlap (>= 3)
    if len(ans_tokens.intersection(ctx_tokens)) >= 3:
        return True

    # Rule 2: strict substring containment
    ans_low = answer.lower()
    for ch in chunks:
        if ans_low in ch.lower():
            return True

    # Rule 3: phone numbers
    import re
    answer_phones = re.findall(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b", ans_low)
    context_phones = re.findall(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b", ctx_text)
    if any(p in context_phones for p in answer_phones):
        return True

    return False


# ---------------------------------------------------------
# Main Generator
# ---------------------------------------------------------
def generate_answer(question: str, chunks: list[str]) -> str:
    if not chunks:
        return "I don't know."

    prompt = build_prompt(question, chunks)

    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=130,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.05,
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = extract_answer(full_output)

    # Final hallucination check
    if not hybrid_grounding(answer, chunks):
        return "I don't know."

    return answer.strip()


# ---------------------------------------------------------
# Manual Micro-Test
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_chunks = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "We will block the card from future use and issue you a new card."
    ]
    q = "How do I report a lost credit card?"
    print("Answer:", generate_answer(q, sample_chunks))
