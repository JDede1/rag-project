"""
main.py — FastAPI Backend for Production RAG (Cloud Run + ONNX)

Features:
    • FAISS + ONNX MPNet semantic retrieval
    • Strict literal-mode generator (Phi or Groq)
    • Strict topic matching 
    • Domain-specific intent filtering 
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
GEN_MODE = os.getenv("GEN_MODE", "local")
CIT_PATTERN = re.compile(r"CIT:(\d+)")


# ---------------------------------------------------------
# FastAPI Initialization
# ---------------------------------------------------------
app = FastAPI(
    title="Fintech RAG API",
    description="Strict Literal RBC Banking RAG using FAISS + ONNX MPNet + Phi/Groq",
    version="11.0.0",
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
# Intent detection (Option B)
# ---------------------------------------------------------
def _intent(chunk: str) -> str:
    """Assign a simple intent label to each chunk."""
    c = chunk.lower()

    if "fraud" in c or "fraudulent" in c:
        return "fraud"
    if "lost" in c or "stolen" in c:
        return "lost"
    if "cancel" in c or "cancelling" in c:
        return "cancel"

    return "general"


# ---------------------------------------------------------
# Retrieval Post-Processing (Strict + Consistent)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.32, max_items: int = 4):
    """
    Updated to Option B:
    1. Sort by final score
    2. Filter out low-scoring chunks
    3. Detect dominant semantic intent
    4. Keep only chunks from the same intent family
    """

    if not results:
        return []

    # 1. Sort
    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    # 2. Strong chunks only
    strong = [
        r for r in ordered
        if r.get("final_score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    if not strong:
        return []

    # 3. Intent classification on strong chunks
    intents = [_intent(r["chunk"]) for r in strong]
    dominant_intent = intents[0]  # top-1 intent = dominant intent

    # 4. Keep chunks belonging to the dominant intent
    filtered = [r for r in strong if _intent(r["chunk"]) == dominant_intent]

    # Safety net
    if not filtered:
        filtered = strong

    return [r["chunk"].strip() for r in filtered][:max_items]


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
        "retriever_model": "onnx-mpnet",
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
    query: str = Query(..., description="User's RBC banking question"),
    top_k: int = Query(5, ge=1, le=10),
):
    start = time.time()

    try:
        # 1. RETRIEVAL
        retrieval_results = retriever.search(query, top_k=top_k)

        # 2. CLEANED CONTEXT (Option B + strict literal)
        clean_chunks = clean_retrieval(retrieval_results)

        if not clean_chunks:
            latency = (time.time() - start) * 1000
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
                latency_ms=latency,
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
                "latency_ms": latency,
            }

        # 3. GENERATE STRICT LITERAL ANSWER
        answer, grounding = generate_answer(query, clean_chunks)

        grounding_score = grounding.get("grounding_score", 0.0)
        context_overlap = grounding.get("context_overlap", 0.0)

        # 4. Extract citations
        citations = sorted({int(m.group(1)) for m in CIT_PATTERN.finditer(answer)})

        # 5. Retriever confidence
        confidence = retrieval_results[0].get("confidence", 0.0)

        latency = (time.time() - start) * 1000

        # 6. Log event
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

        # 7. Return
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
        latency = (time.time() - start) * 1000
        safe = "I don't know."

        log_rag_event(
            query=query,
            answer=safe,
            retrieved=[],
            used_chunks=[],
            citations=[],
            grounding_score=0.0,
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
            "grounding_score": 0.0,
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
