# === Base image ===
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# --------------------------------------------------------
# Install minimal system dependencies (FAISS + ONNX)
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------
# Install Python dependencies
# --------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# Copy project source code
# --------------------------------------------------------
COPY . .

# --------------------------------------------------------
# Copy FAISS index + metadata
# --------------------------------------------------------
COPY data/index /app/data/index

# --------------------------------------------------------
# Copy ONNX model + tokenizer
# --------------------------------------------------------
COPY models/mpnet /app/models/mpnet

# --------------------------------------------------------
# Create logs folder
# --------------------------------------------------------
RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
