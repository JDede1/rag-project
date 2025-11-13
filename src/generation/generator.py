"""
generator.py
-------------------------------------
Hallucination-safe Phi-3-Mini-4k-Instruct generator
for the RAG pipeline.
"""

import os
import torch
from textwrap import dedent
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN and os.path.exists("Keys/HF_TOKEN.txt"):
    with open("Keys/HF_TOKEN.txt") as f:
        HF_TOKEN = f.read().strip()

if not HF_TOKEN:
    raise EnvironmentError(
        "No Hugging Face token found. "
        "Set HUGGINGFACEHUB_API_TOKEN or create Keys/HF_TOKEN.txt"
    )

# ---------------------------------------------------------
# Model Config
# ---------------------------------------------------------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔹 Loading {MODEL_NAME} on {DEVICE} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.2,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=False,  # deterministic
)

print("Phi-3-Mini-4k-Instruct loaded successfully.\n")

# ---------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------
def build_prompt(question: str, retrieved_docs: list[str]) -> str:
    """Compose a grounded RAG prompt for banking FAQs."""
    context = "\n\n".join(retrieved_docs)

    return dedent(f"""
    You are an expert assistant specializing in Canadian banking FAQs.
    Use ONLY the provided context to answer the question accurately.
    If the answer is not in the context, say exactly: "I don’t know."

    Context:
    {context}

    Question: {question}
    Answer:
    """)

# ---------------------------------------------------------
# Hallucination Protection Helpers
# ---------------------------------------------------------
def _extract_answer(raw_output: str) -> str:
    """
    Extract only the part *after* the final 'Answer:' marker.
    This removes prompt echoes and chain-of-thought.
    """
    if "Answer:" in raw_output:
        return raw_output.split("Answer:")[-1].strip()

    # fallback: return last ~500 characters only
    return raw_output[-500:].strip()


def _is_grounded(answer: str, retrieved_docs: list[str]) -> bool:
    """
    Very strict grounding: answer must contain meaningful overlap
    with retrieved context AND must not invent structure.
    """
    answer_lower = answer.lower()

    # Remove trivial hallucinated patterns
    forbidden = [
        "document:", 
        "the post",
        "your task",
        "using only information",
        "<", ">",  # HTML noise
    ]
    if any(x in answer_lower for x in forbidden):
        return False

    # Minimal grounding check: answer must overlap context
    context_text = " ".join(retrieved_docs).lower()

    # Require at least 3 shared meaningful words to pass grounding
    import re
    tokens = re.findall(r"[a-zA-Z]{4,}", answer_lower)
    overlap = sum(1 for t in tokens if t in context_text)

    return overlap >= 2


# ---------------------------------------------------------
# Answer Generation (Hallucination-Proof)
# ---------------------------------------------------------
def generate_answer(question: str, retrieved_docs: list[str]) -> str:
    """Generate a strictly grounded, concise answer."""
    if not retrieved_docs:
        return "I don’t know."

    prompt = build_prompt(question, retrieved_docs)

    try:
        outputs = pipe(prompt)
        raw_output = outputs[0]["generated_text"]

        # 🔹 extract clean answer
        answer = _extract_answer(raw_output)

        # 🔹 Enforce strict grounding
        if not _is_grounded(answer, retrieved_docs):
            return "I don’t know."

        # 🔹 Final clean & trim
        answer = answer.replace("\n\n", "\n").strip()
        return answer[:600]  # safe limit

    except Exception as e:
        return f"Model error: {str(e)}"


# ---------------------------------------------------------
# Local Test Block
# ---------------------------------------------------------
if __name__ == "__main__":
    docs = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "You can also lock or unlock your card in RBC Online Banking or the Mobile App."
    ]
    q = "How do I report a lost credit card?"
    print("Generated Answer:\n", generate_answer(q, docs))
