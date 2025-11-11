"""
generator.py
-------------------------------------
Phi-3-Mini-4k-Instruct generator for the RAG pipeline.

Exports:
    - generate_answer(question, retrieved_docs)
"""

import os
import torch
from textwrap import dedent
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------------------------------------
# 1️⃣  Authentication
# ---------------------------------------------------------
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN and os.path.exists("Keys/HF_TOKEN.txt"):
    with open("Keys/HF_TOKEN.txt") as f:
        HF_TOKEN = f.read().strip()

if not HF_TOKEN:
    raise EnvironmentError(
        "❌ No Hugging Face token found. "
        "Set HUGGINGFACEHUB_API_TOKEN or create Keys/HF_TOKEN.txt"
    )

# ---------------------------------------------------------
# 2️⃣  Model Config (Swapped to Phi-3-mini)
# ---------------------------------------------------------
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔹 Loading {MODEL_NAME} on {DEVICE} ...")

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

# ✅ Load model efficiently for CPU / small GPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

# ✅ Text-generation pipeline (optimized for factual responses)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,          # smaller context = faster response
    temperature=0.2,             # stable and factual
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=False,
    device=0 if DEVICE == "cuda" else -1,
)

print("✅ Phi-3-Mini-4k-Instruct loaded successfully.\n")

# ---------------------------------------------------------
# 3️⃣  Prompt Template
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
# 4️⃣  Answer Generation
# ---------------------------------------------------------
def generate_answer(question: str, retrieved_docs: list[str]) -> str:
    """Generate a grounded, concise answer."""
    if not retrieved_docs:
        return "I don’t know."

    prompt = build_prompt(question, retrieved_docs)

    try:
        outputs = pipe(prompt)
        text = outputs[0]["generated_text"]
        answer = text.split("Answer:", 1)[-1].strip() if "Answer:" in text else text.strip()
        return answer[:800].strip()
    except Exception as e:
        return f"⚠️ Model error: {str(e)}"

# ---------------------------------------------------------
# 5️⃣  Local Test Block
# ---------------------------------------------------------
if __name__ == "__main__":
    docs = [
        "If your RBC credit card is lost or stolen, call 1-800-769-2512 immediately.",
        "You can also lock or unlock your card in RBC Online Banking or the Mobile App."
    ]
    q = "How do I report a lost credit card?"
    print("🧠 Generated Answer:\n", generate_answer(q, docs))
