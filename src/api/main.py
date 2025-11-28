"""
main.py — FastAPI Backend for Production RAG (Cloud Run + ONNX)

Features:
    • FAISS + ONNX MPNet semantic retrieval
    • Strict literal-mode generator (Phi or Groq)
    • Strict topic matching (Option A)
    • Strict grounding enforcement
    • JSONL monitoring logs
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

# Allow src/ imports when running via uvicorn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer
from monitoring.rag_logger import log_rag_event


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
GEN_MODE = os.getenv("GEN_MODE", "local")  # "local" or "groq"

# Extract inline citations from generator output
CIT_PATTERN = re.compile(r"CIT:(\d+)")


# ---------------------------------------------------------
# FastAPI Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Fintech RAG API",
    description="Strict Literal RBC Banking RAG using FAISS + ONNX MPNet + Phi/Groq LLMs",
    version="10.0.0",
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
# Load Retriever
# ---------------------------------------------------------
print("Loading retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# Retrieval Post-Processing (Strict Literal Mode)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.32, max_items: int = 4):
    """
    Returns only the cleaned text chunks.
    Generator assigns its own CIT:1, CIT:2… so retriever IDs are not reused.
    """
    if not results:
        return []

    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    strong = [
        r for r in ordered
        if r.get("final_score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    chunks = [r["chunk"].strip() for r in strong][:max_items]
    return chunks


# ---------------------------------------------------------
# Landing Page
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
def landing(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "generator_mode": GEN_MODE,
        "retriever_model": "onnx-mpnet",
        "index_size": retriever.index.ntotal,
        "embedding_dim": retriever.index.d,
        "record_count": len(retriever.metadata),
        "logging_enabled": True,
    }


# ---------------------------------------------------------
# MAIN RAG ENDPOINT — strict literal + strict topic match
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's RBC banking question"),
    top_k: int = Query(5, ge=1, le=10),
):
    start_time = time.time()

    try:
        # -------------------------------------------------
        # 1. RETRIEVAL
        # -------------------------------------------------
        retrieval_results = retriever.search(query, top_k=top_k)

        # -------------------------------------------------
        # 2. CLEAN RETRIEVAL → ONLY CHUNKS
        # -------------------------------------------------
        clean_chunks = clean_retrieval(retrieval_results)

        # No chunks → guaranteed "I don't know."
        if not clean_chunks:
            latency_ms = (time.time() - start_time) * 1000.0
            safe = "I don't know."

            log_rag_event(
                query=query,
                answer=safe,
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
                "answer": safe,
                "citations_used": [],
                "retrieved": retrieval_results,
                "used_context": [],
                "confidence": 0.0,
                "grounding_score": 0.0,
                "context_overlap": 0.0,
                "latency_ms": latency_ms,
            }

        # -------------------------------------------------
        # 3. GENERATION (strict literal + strict topic match)
        # -------------------------------------------------
        answer, grounding_info = generate_answer(query, clean_chunks)

        grounding_score = grounding_info.get("grounding_score", 0.0)
        context_overlap = grounding_info.get("context_overlap", 0.0)

        # -------------------------------------------------
        # 4. CITATION EXTRACTION
        # -------------------------------------------------
        citations_used = sorted({int(m.group(1)) for m in CIT_PATTERN.finditer(answer)})

        # -------------------------------------------------
        # 5. CONFIDENCE SCORE
        # -------------------------------------------------
        confidence = retrieval_results[0].get("confidence", 0.0) if retrieval_results else 0.0

        # -------------------------------------------------
        # 6. LATENCY
        # -------------------------------------------------
        latency_ms = (time.time() - start_time) * 1000.0

        # -------------------------------------------------
        # 7. LOG EVENT
        # -------------------------------------------------
        log_rag_event(
            query=query,
            answer=answer,
            retrieved=retrieval_results,
            used_chunks=clean_chunks,
            citations=citations_used,
            grounding_score=grounding_score,
            context_overlap=context_overlap,
            confidence=confidence,
            latency_ms=latency_ms,
        )

        # -------------------------------------------------
        # 8. RESPONSE
        # -------------------------------------------------
        return {
            "query": query,
            "answer": answer,
            "citations_used": citations_used,
            "retrieved": retrieval_results,
            "used_context": clean_chunks,
            "confidence": confidence,
            "grounding_score": grounding_score,
            "context_overlap": context_overlap,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000.0
        safe = "I don't know."

        # Log failure event
        log_rag_event(
            query=query,
            answer=safe,
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
            "answer": safe,
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
        port=int(os.getenv("PORT", 8000)),
    )
