# ============================================================
# Base Image
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
# (Pinned versions to avoid Cloud Run crashes)
# ============================================================
COPY requirements.txt .

# --- Install CPU-only torch FIRST (pinned) ---
RUN pip install --no-cache-dir \
    torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# --- Install the remaining packages ---
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy Application Code
# ============================================================
COPY . .

# ============================================================
# Prepare Logs Directory
# ============================================================
RUN mkdir -p /app/logs

# ============================================================
# Expose FastAPI Port
# ============================================================
EXPOSE 8000

# ============================================================
# Start API Server
# ============================================================
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
