from fastapi import FastAPI

app = FastAPI(title="RAG Banking Assistant")

@app.get("/health")
def health_check():
    """
    Simple health check endpoint to verify API is running.
    """
    return {"status": "ok", "message": "RAG API is live 🚀"}
