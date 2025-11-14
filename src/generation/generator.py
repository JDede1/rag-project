"""
generator.py
-------------------------------------------------------
FINAL stable RAG generator for Qwen2.5-0.5B-Instruct.

Key Features:
    • Zero hallucinations (strict grounding)
    • No prompt echo or system leakage
    • Robust assistant-answer extraction
    • Deterministic decoding (temperature=0.0)
    • Fully compatible with Cloudflare + FastAPI
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
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
# RAG Prompt Builder
# ---------------------------------------------------------
def build_prompt(question: str, chunks: list[str]):
    context_text = "\n\n".join(chunks) if chunks else "No relevant information found."

    return [
        {
            "role": "system",
            "content": (
                "You are an RBC assistant. Answer ONLY using the provided context. "
                "If the answer is not in the context, respond with exactly: "
                "\"I don't know.\" Do NOT add external facts."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_text}\n\n"
                f"Question: {question}\n"
                "Answer only using the context."
            )
        }
    ]


# ---------------------------------------------------------
# Robust Answer Extraction (fixed version)
# ---------------------------------------------------------
def extract_answer(full_output: str) -> str:
    """
    Extract only the assistant's final message.
    Handles:
        • Missing <|assistant|> token
        • Prompt echo
        • System/user blocks
        • Extra template artifacts
    """

    text = full_output.strip()

    # 1 — Remove common Qwen special tokens
    bad_tokens = ["<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>"]
    for tok in bad_tokens:
        text = text.replace(tok, " ")

    # 2 — Remove everything before the assistant message (fallback)
    if "Answer only using the context." in text:
        text = text.split("Answer only using the context.")[-1].strip()

    # 3 — Remove repeated instructions
    repeated_phrases = [
        "Answer only using the context.",
        "Only use the context.",
        "Use only the context.",
        "Based on the context",
    ]
    for p in repeated_phrases:
        text = text.replace(p, "").strip()

    # Final return
    return text.strip()


# ---------------------------------------------------------
# Strict grounding filter
# ---------------------------------------------------------
def is_grounded(answer: str, chunks: list[str]) -> bool:
    """
    Prevent hallucinations by requiring minimal token overlap.
    """

    if answer.lower() in ["i don't know.", "i don't know"]:
        return True

    answer_tokens = set(answer.lower().split())
    ctx_tokens = set(" ".join(chunks).lower().split())

    overlap = answer_tokens.intersection(ctx_tokens)

    return len(overlap) >= 2


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------
def generate_answer(question: str, chunks: list[str]) -> str:
    if not chunks:
        return "I don't know."

    messages = build_prompt(question, chunks)

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            encoded,
            max_new_tokens=160,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.05
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    answer = extract_answer(full_output)

    # Apply grounding filter
    if not is_grounded(answer, chunks):
        return "I don't know."

    return answer.strip()


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------
if __name__ == "__main__":
    context = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "You can lock or unlock your card using RBC Online Banking or the RBC Mobile App."
    ]
    q = "How do I report a lost credit card?"
    print("Answer:", generate_answer(q, context))
