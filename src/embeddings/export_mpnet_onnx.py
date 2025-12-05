"""
MPNet → ONNX Export Script
Exports:
    mpnet.onnx
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    config.json
"""

from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

print("\n==========================================")
print("     PHASE 3.5 — MPNet → ONNX Export")
print("==========================================\n")

PROJECT_ROOT = Path("/content/rag-project")
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

onnx_model_path = INDEX_DIR / "mpnet.onnx"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print(f"Loading HF MPNet model: {MODEL_NAME}")

# Use HF AutoModel, not SentenceTransformer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# Save tokenizer files
print("Saving tokenizer files...")
tokenizer.save_pretrained(INDEX_DIR)

# Dummy input for tracing
dummy = tokenizer(
    ["Dummy input for ONNX export"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
)

# Export model
print(f"Exporting ONNX model to: {onnx_model_path}")

torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    onnx_model_path.as_posix(),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    },
    opset_version=18,   # Use modern opset
)

print("\nONNX Export Complete:")
print(f"  - {onnx_model_path}")
print("  - tokenizer.json")
print("  - tokenizer_config.json")
print("  - special_tokens_map.json")
print("  - config.json")

print("\nPHASE 3.5 COMPLETE — ONNX model ready for Cloud Run\n")
