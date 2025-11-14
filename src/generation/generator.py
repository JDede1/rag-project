"""
generator.py
-------------------------------------------------------
Grounded answer generator for RAG using Qwen2.5-0.5B-Instruct.

Why this model?
    • Extremely fast (~0.5 sec inference on CPU/GPU)
    • Perfect for rewriting retrieved chunks in RAG
    • Zero Cloudflare timeout issues
    • Strong instruction following
    • Very low hallucination rate

Grounding features:
    • Answers must rely strictly on provided context
    • If context does not contain the answer, return:
      "I don't know."
    • Deterministic decoding (temperature=0.0)
    • Clean, safe output extraction
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading {MODEL_NAME} on {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto" if torch.cuda.is_available() else None,
)


# ---------------------------------------------------------
# Prompt Construction (Chat format)
# ---------------------------------------------------------

def build_messages(question: str, retrieved_docs: list[str]):
    """
    Construct strict RAG prompt for Qwen2.5.
    """

    context_text = "\n\n".join(retrieved_docs)

    system_msg = (
        "You are an assistant that answers ONLY using the provided context. "
        "If the context does not contain the answer, respond with exactly: "
        "\"I don't know.\" "
        "Do not add external facts. Be concise and factual."
    )

    user_msg = (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer only using the context."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------
# Output Cleaning
# ---------------------------------------------------------

def clean_answer(text: str) -> str:
    text = text.strip()

    # Remove unnecessary prefaces Qwen sometimes generates
    prefixes = [
        "Answer:", "A:", "Here is the answer:", "Response:",
        "Sure,", "Certainly,", "The answer is"
    ]

    lowered = text.lower()
    for p in prefixes:
        if lowered.startswith(p.lower()):
            text = text[len(p):].strip()

    return text[:600].strip()


def is_grounded(answer: str, retrieved_docs: list[str]) -> bool:
    """
    Enforce strict grounding using token overlap filtering.
    """

    if answer.lower() == "i don't know.":
        return True
    if answer.lower() == "i don't know":
        return True

    answer_tokens = set(answer.lower().split())
    context_tokens = set(" ".join(retrieved_docs).lower().split())

    overlap = len(answer_tokens.intersection(context_tokens))

    # Require minimal overlap to avoid hallucination
    return overlap >= 2


# ---------------------------------------------------------
# Main Answer Function
# ---------------------------------------------------------

def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    """
    Generate an answer strictly from retrieved context.
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
            max_new_tokens=180,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.05,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract last assistant response
    if "assistant" in generated_text:
        generated_text = generated_text.split("assistant")[-1].strip()

    answer = clean_answer(generated_text)

    # Enforce grounding
    if not is_grounded(answer, retrieved_chunks):
        return "I don't know."

    return answer


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------
if __name__ == "__main__":
    retrieved = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "You can lock or unlock your card using RBC Online Banking or the RBC Mobile App."
    ]
    q = "How do I report a lost credit card?"
    print(generate_answer(q, retrieved))
