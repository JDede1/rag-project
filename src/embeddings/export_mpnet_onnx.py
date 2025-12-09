"""
MPNet → ONNX Export Script (Cloud-Run Compatible)
================================================

This script exports an ONNX model fully compatible with the ONNXRuntime
available inside Cloud Run. Cloud Run supports up to **IR version 9**,
so we force an opset that guarantees IR ≤ 9.

Output files saved to data/index/:
    • mpnet.onnx
    • tokenizer.json
    • tokenizer_config.json
    • special_tokens_map.json
    • config.json

Run this AFTER Phase 3 (embeddings + index).
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

ONNX_PATH = INDEX_DIR / "mpnet.onnx"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print("\n==========================================")
print("   MPNet → ONNX Export (Cloud Run Safe)")
print("==========================================\n")

# ------------------------------------------------------------
# 1. Load model + tokenizer
# ------------------------------------------------------------
print(f"Loading HuggingFace MPNet model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print("Saving tokenizer files...")
tokenizer.save_pretrained(INDEX_DIR)

print("Saving config.json...")
model.config.to_json_file(INDEX_DIR / "config.json")

# ------------------------------------------------------------
# 2. Dummy input for tracing
# ------------------------------------------------------------
dummy = tokenizer(
    ["Dummy input for ONNX export"],
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)

# ------------------------------------------------------------
# 3. Export ONNX (opset 12 ensures IR v9 in Cloud Run)
# ------------------------------------------------------------
print(f"Exporting ONNX model → {ONNX_PATH}")

torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    ONNX_PATH.as_posix(),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    },
    opset_version=12,   # Cloud Run safe → IR version 9
)

# ------------------------------------------------------------
# 4. Validate ONNX IR version
# ------------------------------------------------------------
print("Validating generated ONNX IR version...")

model_onnx = onnx.load(ONNX_PATH)
print("  IR version:", model_onnx.ir_version)

if model_onnx.ir_version > 9:
    raise RuntimeError(
        f"ONNX IR version {model_onnx.ir_version} > 9 — Cloud Run will reject this model."
    )

print("  ONNX model is Cloud Run–compatible (IR ≤ 9).")

print("\nONNX Export Complete:")
print(" - mpnet.onnx")
print(" - tokenizer.json")
print(" - tokenizer_config.json")
print(" - special_tokens_map.json")
print(" - config.json")

print("\nExport finished successfully.\n")
