# === Base image ===
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Work directory
WORKDIR /app

# Install minimal system dependencies needed for FAISS & general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (caching layer)
COPY requirements.txt .

# Install Python dependencies
# NOTE:
#   • requirements.txt MUST NOT contain torch or transformers
#   • SentenceTransformer backend stays (uses CPU)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining project files
COPY . .

# Ensure logs directory exists (Phase 7/8 monitoring compatibility)
RUN mkdir -p /app/logs

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI (Cloud Run will inject PORT)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
