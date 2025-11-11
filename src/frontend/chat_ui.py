"""
chat_ui.py
--------------------
Simple web interface for interacting with the RAG API backend.
Serves an HTML chat form connected to FastAPI endpoints.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import httpx

app = FastAPI(title="RBC RAG Chat UI")

# ---------------------------------------------------------
# Static & Templates
# ---------------------------------------------------------
app.mount("/static", StaticFiles(directory="src/frontend/static"), name="static")
templates = Jinja2Templates(directory="src/frontend/templates")

# ---------------------------------------------------------
# Backend API Endpoint
# ---------------------------------------------------------
RAG_API_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "answer": None, "query": None, "loading": False}
    )


@app.post("/ask", response_class=HTMLResponse)
async def ask_rag(request: Request):
    """Send user query to the RAG backend and display answer."""
    form = await request.form()
    query = form.get("query")

    # Immediately show a loading message (re-render with spinner)
    loading_context = {"request": request, "query": query, "answer": None, "loading": True}
    loading_html = templates.get_template("index.html").render(loading_context)

    # ⚡ Start fetching the answer from the backend
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{RAG_API_URL}/ask",
                params={"query": query, "top_k": 3},
                timeout=120.0  # wait up to 2 minutes
            )
            result = resp.json()
            answer = result if isinstance(result, str) else result[0].get("answer", str(result))
    except httpx.ReadTimeout:
        answer = "⏱️ The model took too long to respond. Try simplifying your question."
    except Exception as e:
        answer = f"❌ Error: {e}"

    # Render final answer
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "query": query, "answer": answer, "loading": False},
    )
