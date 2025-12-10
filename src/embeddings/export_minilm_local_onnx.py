"""
Local MiniLM → ONNX Export (IR 10 / Opset 18)
=============================================

This version is for LOCAL USE ONLY (Colab, Dev Machines).
Cloud Run CANNOT load this model because IR=10.

Generated file:
    data/index/minilm_local.onnx

Used by:
    • search_engine.py, when ENCODER_MODE=minilm_local_onnx
"""

from pathlib import Path
import torch
import onnx
from transformers import AutoModel, AutoTokenizer

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = INDEX_DIR / "minilm_local.onnx"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("\n==========================================")
print("   MiniLM → Local ONNX Export (IR 10)")
print("==========================================\n")

# ------------------------------------------------------------
# Load HF Model + Tokenizer
# ------------------------------------------------------------
print(f"Loading MiniLM model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# Save tokenizer files once (if not already present)
tokenizer.save_pretrained(INDEX_DIR)
model.config.to_json_file(INDEX_DIR / "config.json")

# ------------------------------------------------------------
# Dummy input
# ------------------------------------------------------------
dummy_inputs = tokenizer(
    ["This is a dummy sentence to trace the ONNX graph."],
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)

# ------------------------------------------------------------
# Export ONNX
# ------------------------------------------------------------
print(f"Exporting LOCAL ONNX → {ONNX_PATH}\n")

torch.onnx.export(
    model,
    (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
    ONNX_PATH.as_posix(),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    },
    opset_version=18,     # ← IR 10+ (normal ONNX)
    do_constant_folding=True,
)

print("Validating ONNX...\n")
model_onnx = onnx.load(ONNX_PATH)

print(f"ONNX IR version: {model_onnx.ir_version}")
print("This is expected to be > 9 (local-only).")

print("\nLocal ONNX export complete.")
print("minilm_local.onnx saved.\n")
