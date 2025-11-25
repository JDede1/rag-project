# === Base image ===
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Work directory
WORKDIR /app

# --------------------------------------------------------
# Install minimal system dependencies (FAISS + ONNX)
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# Copy requirements first (layer caching)
# --------------------------------------------------------
COPY requirements.txt .

# Install Python dependencies
# Requirements MUST NOT contain torch or transformers
# ONNXRuntime is safe for CPU inference
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# Copy project source code
# --------------------------------------------------------
COPY . .

# --------------------------------------------------------
# Copy FAISS index + embeddings (must exist at runtime)
# --------------------------------------------------------
COPY data/index /app/data/index

# --------------------------------------------------------
# Copy ONNX model + tokenizer
# These come from:
#   /models/mpnet/model.onnx
#   /models/mpnet/config.json
#   /models/mpnet/tokenizer.json
#   /models/mpnet/vocab files
# --------------------------------------------------------
COPY models/mpnet /app/models/mpnet

# --------------------------------------------------------
# Ensure logs directory exists (monitoring)
# --------------------------------------------------------
RUN mkdir -p /app/logs

# --------------------------------------------------------
# Expose FastAPI port for Cloud Run
# --------------------------------------------------------
EXPOSE 8000

# --------------------------------------------------------
# Start FastAPI (Cloud Run will inject PORT)
# --------------------------------------------------------
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
