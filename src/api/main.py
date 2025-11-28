""" 
main.py — FastAPI Backend for RAG (Cloud Run + ONNX)

Features:
    • FAISS + ONNX MPNet semantic retrieval
    • Dual-mode generator:
        - Local Phi-3.5-Mini (Colab)
        - Groq-hosted LLMs (Cloud Run)
    • Strict grounding & fallback logic
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
# Environment variables
# ---------------------------------------------------------
GEN_MODE = os.getenv("GEN_MODE", "local")  # "local" or "groq"

# Regex to extract CIT:x from generator output
CIT_PATTERN = re.compile(r"CIT:(\d+)")


# ---------------------------------------------------------
# FastAPI Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Fintech RAG API",
    description="Retrieval-Augmented Generation using FAISS + ONNX MPNet + Phi/Groq LLMs",
    version="9.0.0",
)

# CORS for Streamlit + Cloudflare Tunnel
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
# Retrieval Post-Processing (Strict Literal Mode)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.32, max_items: int = 4):
    """
    Return ONLY the text chunks.

    Citation IDs from retriever are NOT used anymore because the generator
    attaches CIT:1, CIT:2 ... internally based on chunk order.
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
        "record_count": len(retriever.metadata),
        "retriever_model": "onnx-mpnet",
        "index_size": retriever.index.ntotal,
        "embedding_dim": retriever.index.d,
        "generator_mode": GEN_MODE,
        "logging_enabled": True,
    }


# ---------------------------------------------------------
# Main RAG Endpoint
# ---------------------------------------------------------
@app.get("/ask")
def ask(
    query: str = Query(..., description="User's RBC question"),
    top_k: int = Query(5, ge=1, le=10),
):
    start_time = time.time()

    try:
        # 1. Retrieve raw candidates
        retrieval_results = retriever.search(query, top_k=top_k)

        # 2. Clean retrieval → ONLY chunks
        clean_chunks = clean_retrieval(retrieval_results)

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
                "latency_ms": latency_ms,
            }

        # 3. Generate literal-mode answer
        answer, grounding_info = generate_answer(query, clean_chunks)

        grounding_score = grounding_info.get("grounding_score", 0.0)
        context_overlap = grounding_info.get("context_overlap", 0.0)

        # 4. Extract CIT:x from the answer itself
        citations_used = sorted({int(m.group(1)) for m in CIT_PATTERN.finditer(answer)})

        # 5. Confidence (top retrieval score)
        confidence = (
            retrieval_results[0].get("confidence", 0.0)
            if retrieval_results else 0.0
        )

        # 6. Latency
        latency_ms = (time.time() - start_time) * 1000.0

        # 7. Log
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

        # 8. Response
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
        port=int(os.getenv("PORT", 8000)),
    )
