"""
main.py — FastAPI Backend for Production RAG (PyTorch MPNet + Strict Literal Generator)

Features:
    • PyTorch MPNet retrieval (SentenceTransformers)
    • FAISS index search + stable reranking
    • Strict-literal generator (Phi or Groq)
    • Robust topic matching (Option A)
    • Safe grounding enforcement (Option A)
    • Clean retrieval filtering (no intent/category grouping)
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

# Make sure src/ directory is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Internal modules
from src.retrieval.search_engine import RbcRetriever
from src.generation.generator import generate_answer, grounding_details
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
    description="Strict Literal RBC RAG using FAISS + PyTorch MPNet + Phi/Groq",
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
# Static Files & Templates Setup
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
print("Loading PyTorch MPNet retriever...")
retriever = RbcRetriever()
print("Retriever loaded.\n")


# ---------------------------------------------------------
# Stable Retrieval Filtering (No intent grouping)
# ---------------------------------------------------------
def clean_retrieval(results: list, score_threshold: float = 0.32, max_items: int = 4):
    """
    Stable retrieval filter:

        1. Sort chunks by final_score
        2. Keep strong chunks only
        3. Return top-N chunk texts

    Notes:
        • Does NOT group by intent category.
        • Does NOT discard lost/stolen due to fraud noise.
        • Generator handles topic matching internally (Option A).
    """
    if not results:
        return []

    # Sort by retrieval confidence / final score
    ordered = sorted(results, key=lambda r: r.get("final_score", 0.0), reverse=True)

    strong = [
        r for r in ordered
        if r.get("final_score", 0.0) >= score_threshold
        and isinstance(r.get("chunk"), str)
    ]

    if not strong:
        return []

    return [r["chunk"].strip() for r in strong][:max_items]


# ---------------------------------------------------------
# TOPIC-FOCUSED CONTEXT (small but critical fix)
# ---------------------------------------------------------
def focus_context(query: str, chunks: list) -> list:
    """
    Post-filter retrieval context to keep only chunks
    that match the *dominant intent* of the question.

    This prevents fraud-related chunks from diluting
    lost/stolen answers, etc.
    """
    if not chunks:
        return chunks

    q = query.lower()
    lowered = [c.lower() for c in chunks]

    # LOST / STOLEN
    if "lost" in q or "stolen" in q:
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["lost", "stolen", "permanently lost", "misplaced"])
        ]
        if topical:
            return topical

    # FRAUD / UNAUTHORIZED / DISPUTE
    if any(k in q for k in ["fraud", "unauthorized", "dispute"]):
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["fraud", "unauthorized", "dispute"])
        ]
        if topical:
            return topical

    # INTERAC / E-TRANSFER
    if any(k in q for k in ["interac", "e-transfer", "etransfer", "e transfer"]):
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["interac", "e-transfer", "etransfer", "transfer"])
        ]
        if topical:
            return topical

    # PASSWORD / LOGIN / RESET
    if any(k in q for k in ["password", "login", "reset"]):
        topical = [
            c for c in chunks
            if any(k in c.lower() for k in ["password", "login", "reset", "passcode"])
        ]
        if topical:
            return topical

    # Fallback: keep original chunks
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
        "generator_mode": GEN_MODE,                 # local or groq
        "retriever_model": "mpnet-pytorch",         # label for now
        "index_size": retriever.index.ntotal,       # FAISS vector count
        "embedding_dim": retriever.index.d,         # 768
        "record_count": len(retriever.metadata),    # # of chunks
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
        # -------------------------------------------------
        # 1. Semantic Retrieval (PyTorch MPNet)
        # -------------------------------------------------
        retrieval_results = retriever.search(query, top_k=top_k)

        # -------------------------------------------------
        # 2. Filter & Clean Retrieval
        # -------------------------------------------------
        clean_chunks = clean_retrieval(retrieval_results)

        # 🔎 NEW: focus context by query intent
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
        # 3. Strict-Literal Generation
        # -------------------------------------------------
        answer, grounding = generate_answer(query, clean_chunks)

        grounding_score = grounding.get("grounding_score", 0.0)
        context_overlap = grounding.get("context_overlap", 0.0)

        # -------------------------------------------------
        # 4. Extract citations
        # -------------------------------------------------
        citations = sorted({int(m.group(1)) for m in CIT_PATTERN.finditer(answer)})

        # -------------------------------------------------
        # 5. Retriever confidence
        # -------------------------------------------------
        confidence = retrieval_results[0].get("confidence", 0.0)

        latency = (time.time() - start_time) * 1000

        # -------------------------------------------------
        # 6. Log event
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
        # 7. Return final response
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
