"""
MPNet → ONNX Export Script
==========================

Exports the following to data/index/:
    • mpnet.onnx
    • tokenizer.json
    • tokenizer_config.json
    • special_tokens_map.json
    • config.json

This ONNX model is used ONLY in Cloud Run (DEPLOY_ENV=cloud).
Local mode continues to use the SentenceTransformer MPNet model.

This script must be executed AFTER Phase 3 completes (index built).
"""

from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = INDEX_DIR / "mpnet.onnx"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print("\n==========================================")
print("     MPNet → ONNX Export (Cloud Mode)")
print("==========================================\n")

# ------------------------------------------------------------
# 1. Load HF model + tokenizer
# ------------------------------------------------------------
print(f"Loading HuggingFace MPNet model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # disable dropout

# Save tokenizer files to INDEX_DIR
print("Saving tokenizer files...")
tokenizer.save_pretrained(INDEX_DIR)

# Also save model config.json explicitly
print("Saving config.json...")
model.config.to_json_file(INDEX_DIR / "config.json")

# ------------------------------------------------------------
# 2. Create dummy input for ONNX tracing
# ------------------------------------------------------------
dummy_inputs = tokenizer(
    ["Dummy input for ONNX export"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
)

# ------------------------------------------------------------
# 3. Export ONNX model
# ------------------------------------------------------------
print(f"Exporting ONNX model → {ONNX_PATH}")

torch.onnx.export(
    model,
    (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
    ONNX_PATH.as_posix(),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
        "pooler_output": {0: "batch"},
    },
    opset_version=18,  # stable opset, avoids converter errors
)

print("\nONNX Export Complete:")
print(" - mpnet.onnx")
print(" - tokenizer.json")
print(" - tokenizer_config.json")
print(" - special_tokens_map.json")
print(" - config.json")

print("\nExport process finished successfully.\n")
