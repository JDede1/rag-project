"""
main.py — FastAPI Backend for RAG
-------------------------------------------------------
Phase 7 Final:
    • JSONL request logging (monitoring/rag_logger.py)
    • Latency measurement
    • Grounding metrics from generator.grounding_details()
    • 100% compatible with Phase 6 retrieval + generator
"""

import sys
import os
import time
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Allow src/ imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer, grounding_details

# Logging utility
from monitoring.rag_logger import log_rag_event


# ---------------------------------------------------------
# FastAPI Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Fintech RAG API",
    description="Retrieval-Augmented Generation using FAISS + Phi-3.5-Mini (Phase 7 Monitoring)",
    version="7.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Static & Template Setup
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


# ---------------------------------------------------------
# Load Retriever Once
# ---------------------------------------------------------
print("Loading retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# Clean Retrieval (from Phase 6)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.32, max_items: int = 4):
    if not results:
        return [], []

    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    strong = [
        r for r in ordered
        if r.get("final_score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    chunks = [r["chunk"].strip() for r in strong][:max_items]
    citation_ids = [r["citation_id"] for r in strong][:max_items]

    return chunks, citation_ids


# ---------------------------------------------------------
# HTML Landing Page
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse(
        "landing.html",
        {"request": request}
    )


# ---------------------------------------------------------
# Health Check
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "record_count": len(retriever.metadata),
        "retriever_model": retriever.model_name,
        "generator_model": "Phi-3.5-Mini-Instruct (Phase 7)",
        "logging_enabled": True
    }


# ---------------------------------------------------------
# PHASE 7 — Main RAG Endpoint (Monitoring + Grounding)
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's banking question"),
    top_k: int = Query(5, ge=1, le=10)
):
    start_time = time.time()

    try:
        # Step 1 — retrieval
        retrieval_results = retriever.search(query, top_k=top_k)

        # Step 2 — filter strong matches
        clean_chunks, citation_ids = clean_retrieval(retrieval_results)

        # If no strong chunks, fallback immediately
        if not clean_chunks:
            latency_ms = (time.time() - start_time) * 1000.0

            log_rag_event(
                query=query,
                answer="I don't know.",
                retrieved=retrieval_results,
                used_chunks=[],
                citations=[],
                grounding_score=0.0,
                context_overlap=0.0,
                confidence=0.0,
                latency_ms=latency_ms,
            )

            return {
                "query": query,
                "answer": "I don't know.",
                "citations_used": [],
                "retrieved": retrieval_results,
                "used_context": [],
                "confidence": 0.0,
                "grounding_score": 0.0,
                "context_overlap": 0.0,
            }

        # Step 3 — generate answer
        answer = generate_answer(query, clean_chunks)

        # Step 4 — grounding metrics from generator
        grounding = grounding_details(answer, clean_chunks)
        grounding_score = grounding["grounding_score"]
        context_overlap = grounding["context_overlap"]

        # Step 5 — model confidence (retriever-level)
        confidence = (
            retrieval_results[0].get("confidence", 0.0)
            if retrieval_results else 0.0
        )

        # Step 6 — latency
        latency_ms = (time.time() - start_time) * 1000.0

        # Step 7 — Log everything
        log_rag_event(
            query=query,
            answer=answer,
            retrieved=retrieval_results,
            used_chunks=clean_chunks,
            citations=citation_ids,
            grounding_score=grounding_score,
            context_overlap=context_overlap,
            confidence=confidence,
            latency_ms=latency_ms,
        )

        # Step 8 — return full results
        return {
            "query": query,
            "answer": answer,
            "citations_used": citation_ids,
            "retrieved": retrieval_results,
            "used_context": clean_chunks,
            "confidence": confidence,
            "grounding_score": grounding_score,
            "context_overlap": context_overlap,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000.0

        log_rag_event(
            query=query,
            answer="I don't know.",
            retrieved=[],
            used_chunks=[],
            citations=[],
            grounding_score=0.0,
            context_overlap=0.0,
            confidence=0.0,
            latency_ms=latency_ms,
        )

        return {
            "query": query,
            "answer": "I don't know.",
            "error": str(e),
            "citations_used": [],
            "used_context": [],
            "retrieved": [],
            "confidence": 0.0,
            "grounding_score": 0.0,
            "context_overlap": 0.0,
            "latency_ms": latency_ms,
        }


# ---------------------------------------------------------
# Manual Run
# ---------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )
