"""
main.py — FastAPI Backend for Production RAG (Cloud Run + ONNX)

Features:
    • FAISS + ONNX MPNet retrieval
    • Strict literal-mode generator (Phi or Groq)
    • Option B topic matching (in generator.py)
    • Stable retrieval filtering (no intent-category loss)
    • Strict grounding enforcement
    • JSONL monitoring logs (monitoring/rag_logger.py)
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

# Make sure src/ is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Internal modules
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
    description="Strict Literal RBC RAG using FAISS + ONNX MPNet + Phi/Groq",
    version="12.0.0",
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
# Load Retriever (global)
# ---------------------------------------------------------
print("Loading retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# CORRECT & STABLE RETRIEVAL FILTERING 
# (NO intent filtering — avoids lost/stolen vs fraud misrouting)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.32, max_items: int = 4):
    """
    Stable retrieval filtering:

        1. Sort chunks by final_score
        2. Keep strong chunks only
        3. Return only the top-N chunk texts

    This version does NOT:
        - group by intent category
        - rely on semantic grouping
        - discard lost/stolen chunks because of fraud noise

    All strict topic checking is handled inside generator.py (Option B).
    """
    if not results:
        return []

    # Sort by retrieval strength
    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    # Keep high-quality chunks
    strong = [
        r for r in ordered
        if r.get("final_score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    if not strong:
        return []

    # Return only the chunk strings
    return [r["chunk"].strip() for r in strong][:max_items]


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
    start_time = time.time()

    try:
        # -------------------------------------------------
        # 1. SEMANTIC RETRIEVAL
        # -------------------------------------------------
        retrieval_results = retriever.search(query, top_k=top_k)

        # -------------------------------------------------
        # 2. CLEAN + STABLE RETRIEVAL (NO INTENT FILTERING)
        # -------------------------------------------------
        clean_chunks = clean_retrieval(retrieval_results)

        # No usable context → safe fallback
        if not clean_chunks:
            latency = (time.time() - start_time) * 1000
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

        # -------------------------------------------------
        # 3. GENERATE STRICT-LITERAL ANSWER
        # -------------------------------------------------
        answer, grounding = generate_answer(query, clean_chunks)

        grounding_score = grounding.get("grounding_score", 0.0)
        context_overlap = grounding.get("context_overlap", 0.0)

        # -------------------------------------------------
        # 4. Extract inline citations
        # -------------------------------------------------
        citations = sorted({int(m.group(1)) for m in CIT_PATTERN.finditer(answer)})

        # -------------------------------------------------
        # 5. Retriever confidence
        # -------------------------------------------------
        confidence = retrieval_results[0].get("confidence", 0.0)

        latency = (time.time() - start_time) * 1000

        # -------------------------------------------------
        # 6. Log JSONL event for monitoring
        # -------------------------------------------------
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

        # -------------------------------------------------
        # 7. Return response payload
        # -------------------------------------------------
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
