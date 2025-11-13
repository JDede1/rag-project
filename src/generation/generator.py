"""
generator.py
-------------------------------------------------------
Grounded answer generator for RAG using Phi-3.5-Mini-Instruct.

Upgrades from previous version:
    • Uses microsoft/Phi-3.5-mini-instruct (chat model)
    • Uses proper chat format (system / user messages)
    • Strict grounding: answers must not include facts outside context
    • Deterministic decoding (temperature=0.0)
    • Clean, controlled output extraction
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading {MODEL_NAME} on {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
)


# ---------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------

def build_messages(question: str, retrieved_docs: list[str]):
    """
    Build a structured chat prompt for Phi-3.5.

    Rules:
        • Assistant must answer ONLY using provided context
        • If answer is missing, respond with: "I don't know."
        • Banking-domain specific, factual, concise
    """

    context_block = "\n\n".join(retrieved_docs)

    system_msg = (
        "You are an expert assistant specializing in Canadian banking FAQs. "
        "You must answer strictly based on the provided context. "
        "If the answer is not contained in the context, respond only with: "
        "\"I don't know.\" "
        "Do not add extra details or external knowledge. "
        "Be concise, factual, and avoid speculative statements."
    )

    user_msg = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Provide the answer based ONLY on the context."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------
# Output Cleaning
# ---------------------------------------------------------

def clean_answer(text: str) -> str:
    """Clean model output safely and deterministically."""
    text = text.strip()

    # Remove any model-added preamble
    for prefix in ["Answer:", "A:", "Here is the answer:", "Sure,", "Certainly,"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Restrict output length for safety
    return text[:600].strip()


def is_grounded(answer: str, retrieved_docs: list[str]) -> bool:
    """
    Enforce strict grounding: the answer must match or overlap meaningfully
    with the retrieved context, not invent details.
    """

    if answer.lower() == "i don't know.":
        return True

    answer_tokens = set(answer.lower().split())
    context_tokens = set(" ".join(retrieved_docs).lower().split())

    overlap = len(answer_tokens.intersection(context_tokens))

    # A minimum overlap of 3 tokens prevents hallucination
    return overlap >= 3


# ---------------------------------------------------------
# Main Answer Generator
# ---------------------------------------------------------

def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    """
    Predict answer using Phi-3.5-mini-instruct with strict grounding.
    """
    if not retrieved_chunks:
        return "I don't know."

    messages = build_messages(question, retrieved_chunks)

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=200,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.05,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the assistant's last response
    if "assistant" in generated:
        generated = generated.split("assistant")[-1]

    answer = clean_answer(generated)

    if not is_grounded(answer, retrieved_chunks):
        return "I don't know."

    return answer


# ---------------------------------------------------------
# Manual test block
# ---------------------------------------------------------
if __name__ == "__main__":
    context = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "You can also lock or unlock your card using RBC Online Banking or the RBC Mobile App."
    ]
    q = "How do I report a lost credit card?"
    print("Answer:", generate_answer(q, context))
