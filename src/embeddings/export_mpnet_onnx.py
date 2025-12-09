"""
MPNet → ONNX Export Script (Cloud-Run Compatible)
================================================

Exports a Cloud Run–safe ONNX version of the MPNet encoder used for
retrieval. This ONNX model MUST match the exact encoder used to build
your FAISS index (768-dim MPNet).

Cloud Run requirement:
    - ONNX IR version must be ≤ 9
    - Opset 12 ensures IR ≤ 9

Files saved to data/index/:
    • mpnet.onnx
    • tokenizer.json
    • tokenizer_config.json
    • special_tokens_map.json
    • config.json

Run only AFTER Phase 3 completes.
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
print("       MPNet → ONNX Export (Cloud Safe)")
print("==========================================\n")

# ------------------------------------------------------------
# 1. Load HF model + tokenizer
# ------------------------------------------------------------
print(f"Loading MPNet model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print("Saving tokenizer + config...")
tokenizer.save_pretrained(INDEX_DIR)
model.config.to_json_file(INDEX_DIR / "config.json")

# ------------------------------------------------------------
# 2. Create dummy input for tracing
# ------------------------------------------------------------
dummy_inputs = tokenizer(
    ["Dummy input for ONNX tracing"],
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)

# ------------------------------------------------------------
# 3. Export ONNX (opset=12 → IR ≤ 9)
# ------------------------------------------------------------
print(f"Exporting ONNX → {ONNX_PATH}\n")

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
    opset_version=12,     # Ensures Cloud Run compatible IR v9
)

# ------------------------------------------------------------
# 4. Validate IR version
# ------------------------------------------------------------
print("Validating ONNX IR version...\n")

model_onnx = onnx.load(ONNX_PATH)
ir_version = model_onnx.ir_version

print(f"Detected IR version: {ir_version}")

if ir_version > 9:
    raise RuntimeError(
        f"ERROR: ONNX IR version {ir_version} exceeds Cloud Run limit (≤ 9)."
    )

print("✔️ ONNX model is Cloud Run–compatible (IR ≤ 9)\n")

print("Export complete. Files generated:")
print("  - mpnet.onnx")
print("  - tokenizer.json")
print("  - tokenizer_config.json")
print("  - special_tokens_map.json")
print("  - config.json\n")

print("MPNet ONNX export finished successfully.\n")
