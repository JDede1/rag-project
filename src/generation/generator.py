"""
generator.py
-------------------------------------------------------
Hybrid-grounded RAG generator for Phi-3.5-Mini-Instruct.

Upgrades in this corrected version:
    • Removes system / instruction leakage
    • Cuts everything before/after the "Answer:" section
    • Prevents Phi from echoing the entire prompt
    • Stronger regex-style extraction logic
    • Still deterministic, still hallucination-safe
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
# Prompt Builder for Phi (no chat template)
# ---------------------------------------------------------
def build_prompt(question: str, chunks: list[str]) -> str:
    """
    Phi-3.5 is sensitive to long instructions.
    Using a minimal, tightly controlled prompt prevents leakage.
    """

    context_text = "\n\n".join(chunks) if chunks else "No context available."

    prompt = (
        "You are an assistant that answers ONLY using the provided context.\n"
        "If the answer is not in the context, reply with exactly: I don't know.\n"
        "Do NOT add outside facts.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    return prompt


# ---------------------------------------------------------
# Robust Output Extraction (Corrected)
# ---------------------------------------------------------
def extract_answer(full_output: str) -> str:
    """
    Strict cleaning:
        1. Remove everything before the final 'Answer:'
        2. Drop system/echoed instructions
        3. Prevent leakage of prompt content
    """

    text = full_output.strip()

    # 1 — Keep everything AFTER the last occurrence of "Answer:"
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()

    # 2 — Remove common instruction echoes
    bad_phrases = [
        "You are an assistant",
        "ONLY using the provided context",
        "Do NOT add outside facts",
        "Context:",
        "Question:",
        "Answer only using the context",
        "Use only the context",
        "Give the answer using only the context",
    ]
    for p in bad_phrases:
        text = text.replace(p, "").strip()

    # 3 — Remove any leftover prompt fragments
    for word in ["Context", "Question", "system", "assistant"]:
        if word + ":" in text:
            text = text.split(word + ":")[0].strip()

    # Final cleanup
    return text.strip()


# ---------------------------------------------------------
# Hybrid Grounding (hallucination guard)
# ---------------------------------------------------------
def hybrid_grounding(answer: str, chunks: list[str]) -> bool:
    if not answer or answer.lower() == "i don't know":
        return True

    ans_tokens = set(answer.lower().split())
    ctx_tokens = set(" ".join(chunks).lower().split())

    stopwords = {
        "the", "is", "a", "to", "of", "and", "in",
        "for", "on", "by", "you", "your", "or"
    }
    ans_tokens = ans_tokens - stopwords

    # Rule 1 — token overlap
    if len(ans_tokens.intersection(ctx_tokens)) >= 1:
        return True

    # Rule 2 — soft substring match
    for ch in chunks:
        ch_low = ch.lower()
        if ch_low[:60] in answer.lower():
            return True
        if len(ch_low) > 40 and ch_low[:50] in answer.lower():
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
            max_new_tokens=180,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.05
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    answer = extract_answer(full_output)

    # Hallucination filter
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
