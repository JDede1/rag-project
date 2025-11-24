import json
from datetime import datetime
from pathlib import Path

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
    """Append a single RAG request/response event into a JSONL file."""

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

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    return True
