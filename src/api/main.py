"""
main.py — FastAPI Backend for Production RAG

Fully aligned with:
    • Updated MiniLM embeddings (chunk-only + category hint)
    • Topic-aware hybrid retrieval
    • Strict literal generator
    • Monitoring (rag_logger)
"""

import sys
import os
import time
import re
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Ensure src/ is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Internal modules
from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer
from monitoring.rag_logger import log_rag_event


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
GEN_MODE = os.getenv("GEN_MODE", "local").lower().strip()
CIT_PATTERN = re.compile(r"CIT:(\d+)", re.IGNORECASE)


# ---------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Fintech RAG API",
    description="RBC RAG using FAISS + MiniLM Encoder",
    version="12.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Static Files & Templates
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ---------------------------------------------------------
# Load Retriever (Global)
# ---------------------------------------------------------
print("Loading MiniLM retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# CLEAN RETRIEVAL
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.18, max_items: int = 6):
    if not results:
        return []

    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)
    top = ordered[0]
    top_topic = top.get("topic", "general")

    strong = []
    for r in ordered:
        score = r.get("final_score", 0.0)
        topic = r.get("topic", "general")
        chunk_text = r.get("chunk")

        if not isinstance(chunk_text, str):
            continue

        if score >= score_threshold or topic == top_topic:
            strong.append(chunk_text.strip())

        if len(strong) >= max_items:
            break

    return strong


# ---------------------------------------------------------
# CONTEXT FOCUS
# ---------------------------------------------------------
def focus_context(query: str, chunks: list) -> list:
    if not chunks:
        return chunks

    q = query.lower()

    def keep_if_contains(keywords):
        filtered = [c for c in chunks if any(k in c.lower() for k in keywords)]
        return filtered if filtered else None

    if "lost" in q or "stolen" in q:
        exact = keep_if_contains(["lost", "stolen", "permanently lost", "misplaced"])
        if exact:
            return exact

    if any(k in q for k in ["fraud", "unauthorized", "dispute"]):
        exact = keep_if_contains(["fraud", "unauthorized", "dispute"])
        if exact:
            return exact

    if any(k in q for k in ["password", "login", "reset", "passcode"]):
        exact = keep_if_contains(["password", "login", "reset", "passcode"])
        if exact:
            return exact

    if any(k in q for k in ["interac", "e-transfer", "etransfer", "transfer"]):
        exact = keep_if_contains(["interac", "e-transfer", "etransfer", "transfer"])
        if exact:
            return exact

    return chunks


# ---------------------------------------------------------
# Landing Page
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "generator_mode": GEN_MODE,
        "retriever_model": "minilm",
        "index_size": retriever.index.ntotal,
        "embedding_dim": retriever.index.d,
        "record_count": len(retriever.metadata),
        "logging_enabled": True,
    }


# ---------------------------------------------------------
# MAIN RAG ENDPOINT
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User query for RBC support"),
    top_k: int = Query(5, ge=1, le=10),
):
    start_time = time.time()

    try:
        retrieval_results = retriever.search(query, top_k=top_k)

        clean_chunks = clean_retrieval(retrieval_results)
        clean_chunks = focus_context(query, clean_chunks)

        if not clean_chunks:
            latency = (time.time() - start_time) * 1000
            safe = "I don't know."

            log_rag_event(
                query=query,
                answer=safe,
                retrieved=retrieval_results,
                used_chunks=[],
                citations=[],
                grounding_score=1.0,
                context_overlap=0.0,
                confidence=0.0,
                latency_ms=latency,
            )

            return {
                "query": query,
                "answer": safe,
                "citations_used": [],
                "retrieved": retrieval_results,
                "used_context": [],
                "confidence": 0.0,
                "grounding_score": 1.0,
                "context_overlap": 0.0,
                "latency_ms": latency,
            }

        answer, _ = generate_answer(query, clean_chunks)

        grounding_score = 1.0
        context_overlap = 1.0

        citations = sorted({int(m.group(1)) for m in CIT_PATTERN.finditer(answer)})
        confidence = retrieval_results[0].get("confidence", 0.0)

        latency = (time.time() - start_time) * 1000

        log_rag_event(
            query=query,
            answer=answer,
            retrieved=retrieval_results,
            used_chunks=clean_chunks,
            citations=citations,
            grounding_score=grounding_score,
            context_overlap=context_overlap,
            confidence=confidence,
            latency_ms=latency,
        )

        return {
            "query": query,
            "answer": answer,
            "citations_used": citations,
            "retrieved": retrieval_results,
            "used_context": clean_chunks,
            "confidence": confidence,
            "grounding_score": grounding_score,
            "context_overlap": context_overlap,
            "latency_ms": latency,
        }

    except Exception as e:
        latency = (time.time() - start_time) * 1000
        safe = "I don't know."

        log_rag_event(
            query=query,
            answer=safe,
            retrieved=[],
            used_chunks=[],
            citations=[],
            grounding_score=1.0,
            context_overlap=0.0,
            confidence=0.0,
            latency_ms=latency,
        )

        return {
            "query": query,
            "answer": safe,
            "error": str(e),
            "citations_used": [],
            "used_context": [],
            "retrieved": [],
            "confidence": 0.0,
            "grounding_score": 1.0,
            "context_overlap": 0.0,
            "latency_ms": latency,
        }


# ---------------------------------------------------------
# Manual Run
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
