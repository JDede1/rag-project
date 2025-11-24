import json
from datetime import datetime
from pathlib import Path
import sys

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "rag_requests.jsonl"


def log_rag_event(
    query: str,
    answer: str,
    retrieved: list,
    used_chunks: list,
    citations: list,
    grounding_score: float,
    context_overlap: float,
    confidence: float,
    latency_ms: float,
):
    """Append a single RAG request/response event into a JSONL file and stdout."""

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "grounding_score": grounding_score,
        "context_overlap": context_overlap,
        "latency_ms": latency_ms,
        "retrieved": retrieved,
        "used_chunks": used_chunks,
    }

    # 1) File logging (still works in Colab / local)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        # Don't crash the API if file writing fails
        print(f"[rag_logger] File logging failed: {e}", file=sys.stderr, flush=True)

    # 2) Cloud Run / Cloud Logging: log to stdout as JSON
    print(json.dumps(record), flush=True)

    return True
