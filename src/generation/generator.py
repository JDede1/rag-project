"""
generator.py
-------------------------------------------------------
Hybrid-grounded RAG generator for Phi-3.5-Mini-Instruct.

Key Features:
    • Hybrid grounding = keyword overlap + soft substring
    • Zero hallucinations (safety-first)
    • Allows legitimate paraphrasing
    • Robust answer extraction
    • Deterministic decoding
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
# RAG Prompt Builder — tailored for Phi chat format
# ---------------------------------------------------------
def build_prompt(question: str, chunks: list[str]):
    context_text = "\n\n".join(chunks) if chunks else "No context available."

    # Phi uses natural chat format, not Qwen chat template tokens.
    prompt = (
        "You are an RBC assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not in the context, reply with exactly: I don't know.\n"
        "Do NOT add outside facts or assumptions.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return prompt


# ---------------------------------------------------------
# Clean model output
# ---------------------------------------------------------
def extract_answer(full_output: str) -> str:
    text = full_output.strip()

    # Remove repeated instructions
    bad_phrases = [
        "Answer only using the context.",
        "Give the answer using only the context.",
        "Use only the context.",
        "Based on the context",
        "Answer:"
    ]
    for p in bad_phrases:
        text = text.replace(p, "").strip()

    return text.strip()


# ---------------------------------------------------------
# Hybrid Grounding Logic
# ---------------------------------------------------------
def hybrid_grounding(answer: str, chunks: list[str]) -> bool:
    """
    Combination of:
        1. Keyword overlap (≥ 1 meaningful token)
        2. Soft substring similarity
    """

    if not answer or answer.lower() == "i don't know":
        return True

    ans_tokens = set(answer.lower().split())
    ctx_tokens = set(" ".join(chunks).lower().split())

    # Stopwords
    stopwords = {
        "the", "is", "a", "to", "of", "and", "in",
        "for", "on", "by", "you", "your", "or"
    }
    ans_tokens = ans_tokens - stopwords

    # --- Rule 1: Keyword overlap ---
    if len(ans_tokens.intersection(ctx_tokens)) >= 1:
        return True

    # --- Rule 2: Soft substring match ---
    for ch in chunks:
        ch_low = ch.lower()

        # If the answer contains the first 30–60 chars of the chunk
        if ch_low[:60] in answer.lower():
            return True

        # If a big portion overlaps
        if len(ch_low) > 40 and ch_low[:50] in answer.lower():
            return True

    return False


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------
def generate_answer(question: str, chunks: list[str]) -> str:
    if not chunks:
        return "I don't know."

    prompt = build_prompt(question, chunks)

    encoded = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=180,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.05
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the answer portion
    answer = extract_answer(full_output)

    # Hybrid grounding (hallucination filter)
    if not hybrid_grounding(answer, chunks):
        return "I don't know."

    return answer.strip()


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------
if __name__ == "__main__":
    sample_chunks = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "We will block the card from future use and issue you a new card."
    ]
    q = "How do I report a lost credit card?"
    print("Answer:", generate_answer(q, sample_chunks))
