"""
MiniLM → ONNX Export Script
===========================

Exports the following to data/index/:
    • minilm.onnx
    • tokenizer.json
    • tokenizer_config.json
    • special_tokens_map.json
    • config.json

This ONNX model is used ONLY in Cloud Run (DEPLOY_ENV=cloud).
Local mode continues to use the SentenceTransformer MPNet or MiniLM model.

This script must be executed AFTER Phase 3 completes (index built).
"""

from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import onnx
from onnx import version_converter


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = INDEX_DIR / "minilm.onnx"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("\n==========================================")
print("     MiniLM → ONNX Export (Cloud Mode)")
print("==========================================\n")


# ------------------------------------------------------------
# 1. Load HF model + tokenizer
# ------------------------------------------------------------
print(f"Loading HuggingFace MiniLM model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # disable dropout

# Save tokenizer + config to INDEX_DIR
print("Saving tokenizer + config files...")
tokenizer.save_pretrained(INDEX_DIR)
model.config.to_json_file(INDEX_DIR / "config.json")


# ------------------------------------------------------------
# 2. Create dummy input for ONNX tracing
# ------------------------------------------------------------
dummy_inputs = tokenizer(
    ["Dummy ONNX export input"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=256,
)


# ------------------------------------------------------------
# 3. Export ONNX model with opset 12 (IR 7 or 8)
# ------------------------------------------------------------
print(f"Exporting ONNX model to {ONNX_PATH}")

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
    opset_version=12,   # Cloud Run compatible
)


# ------------------------------------------------------------
# 4. Validate IR version is safe (≤ 9)
# ------------------------------------------------------------
print("\nValidating ONNX IR version...")

model_proto = onnx.load(ONNX_PATH.as_posix())

ir_version = model_proto.ir_version
print(f"Detected IR version: {ir_version}")

if ir_version > 9:
    raise RuntimeError(
        f"INVALID IR VERSION: {ir_version}\n"
        "Cloud Run only supports IR ≤ 9.\n"
        "MiniLM export must use opset_version=12 to keep IR within limits."
    )

print("IR version OK — Cloud Run compatible.\n")


# ------------------------------------------------------------
# 5. Final confirmation
# ------------------------------------------------------------
print("==========================================")
print(" ONNX Export Complete — Files generated:")
print("  - minilm.onnx")
print("  - tokenizer.json")
print("  - tokenizer_config.json")
print("  - special_tokens_map.json")
print("  - config.json")
print("==========================================\n")

print("Export completed successfully.\n")
