# ============================================================
# Base Image
# ============================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ============================================================
# System Dependencies (FAISS + ONNX)
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Install Python Dependencies
# ============================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy Full Application Source Code
# ============================================================
COPY . .

# ============================================================
# Ensure Logs Directory Exists
# ============================================================
RUN mkdir -p /app/logs

# ============================================================
# Expose FastAPI Port
# ============================================================
EXPOSE 8000

# ============================================================
# Start Server
# ============================================================
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
