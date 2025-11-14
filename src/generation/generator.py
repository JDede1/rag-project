"""
generator.py
-------------------------------------------------------
FINAL stable RAG generator for Qwen2.5-0.5B-Instruct.
Fixes:
    • No prompt echo
    • No system-message leakage
    • Returns ONLY assistant answer
    • Uses correct chat template
    • Clean deterministic decoding
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"[Generator] Loading {MODEL_NAME} on {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=DTYPE,
)


# ---------------------------------------------------------
# Build RAG Prompt
# ---------------------------------------------------------
def build_prompt(question: str, chunks: list[str]):
    if not chunks:
        context_text = "No relevant information found."
    else:
        context_text = "\n\n".join(chunks)

    return [
        {
            "role": "system",
            "content": (
                "You are an RBC assistant. Answer ONLY using the context. "
                "If the answer is not in the context, respond with exactly: "
                "\"I don't know.\""
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
# Clean answer from Qwen formatting
# ---------------------------------------------------------
def extract_answer(full_output: str) -> str:
    """
    Extract only the assistant's final message after applying chat template.
    """

    # Qwen templates produce this pattern:
    # <s>system ...</s><s>user ...</s><s>assistant ...</s>
    if "<|assistant|>" in full_output:
        return full_output.split("<|assistant|>")[-1].strip()

    # Fallback: remove everything before last user message
    if "Answer only using the context." in full_output:
        return full_output.split("Answer only using the context.")[-1].strip()

    return full_output.strip()


# ---------------------------------------------------------
# Strict grounding check
# ---------------------------------------------------------
def is_grounded(answer: str, chunks: list[str]) -> bool:
    if answer.lower().strip() in ["i don't know.", "i don't know"]:
        return True

    answer_tokens = set(answer.lower().split())
    context_tokens = set(" ".join(chunks).lower().split())

    return len(answer_tokens.intersection(context_tokens)) >= 2


# ---------------------------------------------------------
# Main Generation Function
# ---------------------------------------------------------
def generate_answer(question: str, chunks: list[str]) -> str:
    if not chunks:
        return "I don't know."

    messages = build_prompt(question, chunks)

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,            # IMPORTANT: required for correct model behavior
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            model_inputs,
            max_new_tokens=150,
            temperature=0.0,
            top_p=1.0,
            do_sample=False
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = extract_answer(full_output)

    # enforce grounding
    if not is_grounded(answer, chunks):
        return "I don't know."

    # final cleanup
    return answer.strip()


# ---------------------------------------------------------
# Manual Test
# ---------------------------------------------------------
if __name__ == "__main__":
    test_context = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "You can lock or unlock your card using RBC Online Banking or the RBC Mobile App."
    ]
    question = "How do I report a lost credit card?"
    print(generate_answer(question, test_context))
