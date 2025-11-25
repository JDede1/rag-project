"""
convert_mpnet_to_onnx.py
-------------------------------------------------------
Convert the SentenceTransformer model
    sentence-transformers/all-mpnet-base-v2
into ONNX format for fast, CPU-optimized inference
inside Google Cloud Run.

This script outputs:
    models/mpnet/model.onnx
    models/mpnet/tokenizer.json
    models/mpnet/config.json
    models/mpnet/special_tokens_map.json

Run this ONCE in Colab:

    !python src/retrieval/convert_mpnet_to_onnx.py

Cloud Run will load these files using onnxruntime.
"""

from pathlib import Path
import os
import torch
from transformers import AutoTokenizer, AutoModel
import onnx
import onnxruntime as ort


# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "mpnet"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


# ============================================================
# Export Function
# ============================================================
def export_mpnet_to_onnx():
    print("==========================================")
    print(" Loading model + tokenizer")
    print("==========================================")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Save tokenizer files for runtime
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Tokenizer saved → {MODEL_DIR}")

    print("==========================================")
    print(" Preparing dummy input")
    print("==========================================")

    sample = tokenizer(
        "This is a dummy ONNX export input",
        return_tensors="pt"
    )

    # ONNX output path
    onnx_path = MODEL_DIR / "model.onnx"

    print("==========================================")
    print(" Exporting to ONNX (may take ~15–20s)")
    print("==========================================")

    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=14,
    )

    print(f"ONNX model saved → {onnx_path}")

    print("==========================================")
    print(" Validating ONNX model")
    print("==========================================")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print("==========================================")
    print(" SUCCESS: Model converted to ONNX!")
    print("==========================================")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    export_mpnet_to_onnx()
