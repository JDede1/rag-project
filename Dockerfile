# ============================================================
# Base Image (Slim, Cloud Run Friendly)
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ============================================================
# System Dependencies (FAISS + Torch CPU)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install Python Dependencies
# ============================================================
COPY requirements.txt .

# 1. Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# 2. Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy Application Code
# ============================================================
COPY . .

# ============================================================
# FORCE INCLUDE FAISS + ONNX + EMBEDDINGS
# ============================================================
RUN mkdir -p /app/data/index

# FAISS + metadata
COPY data/index/rbc_faiss.index        /app/data/index/
COPY data/index/rbc_embeddings.npy     /app/data/index/
COPY data/index/rbc_metadata.parquet   /app/data/index/

# ============================================================
# MiniLM ONNX (Cloud Run Encoder)
# ============================================================
COPY data/index/minilm.onnx            /app/data/index/

# Tokenizer files (MiniLM)
COPY data/index/config.json            /app/data/index/
COPY data/index/tokenizer.json         /app/data/index/
COPY data/index/tokenizer_config.json  /app/data/index/
COPY data/index/special_tokens_map.json /app/data/index/

# ============================================================
# Prepare Logs Directory
# ============================================================
RUN mkdir -p /app/logs

# ============================================================
# Cloud Run Environment Variables
# ============================================================
ENV DEPLOY_ENV=cloud
ENV GEN_MODE=groq
ENV ENFORCE_GROUNDING=true

# ============================================================
# Expose FastAPI Port
# ============================================================
EXPOSE 8000

# ============================================================
# Start API Server
# ============================================================
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
