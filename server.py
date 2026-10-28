"""
NeuralDocs web server.

Run:  uvicorn server:app --reload     then open http://localhost:8000
"""

import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import rag

MAX_UPLOAD_MB = 25

@asynccontextmanager
async def lifespan(app):
    # index the sample PDFs in the background so the server answers health checks right away
    threading.Thread(target=rag.load_samples, daemon=True).start()
    yield


app = FastAPI(title="NeuralDocs", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)


@app.middleware("http")
async def revalidate_static(request, call_next):
    # browsers re-check the interface on each load (a cheap 304 when unchanged), so updates show up at once
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.setdefault("Cache-Control", "no-cache")
    return response


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=12000)


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = Field(default_factory=list, max_length=40)
    source: str | None = None
    mode: Literal["auto", "fast", "think"] = "auto"


@app.get("/api/status")
def status():
    docs = rag.list_documents()
    return {
        "ready": rag.groq_client is not None,
        "documents": docs,
        "chunks": sum(d["chunks"] for d in docs),
        "models": rag.MODELS,
    }


@app.get("/api/documents")
def documents():
    return rag.list_documents()


@app.post("/api/documents")
def upload(files: list[UploadFile] = File(...)):
    # a plain def runs in FastAPI's thread pool, so embedding never blocks other requests
    results = []
    for f in files:
        name = os.path.basename(f.filename or "document.pdf")
        if not name.lower().endswith(".pdf"):
            results.append({"name": name, "ok": False, "error": "Only PDF files are supported."})
            continue
        data = f.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            results.append({"name": name, "ok": False, "error": f"Larger than {MAX_UPLOAD_MB} MB."})
            continue
        try:
            n = rag.ingest_pdf(data, name)
        except Exception as e:
            results.append({"name": name, "ok": False, "error": f"Could not read this PDF ({type(e).__name__})."})
            continue
        if n:
            results.append({"name": name, "ok": True, "chunks": n})
        else:
            results.append({"name": name, "ok": False, "error": "No text found. It may be a scanned image."})
    return {"results": results, "documents": rag.list_documents()}


@app.delete("/api/documents/{name}")
def delete(name: str):
    if name in rag.sample_names():
        raise HTTPException(403, "This sample document is part of the demo and can't be removed.")
    if not rag.delete_document(name):
        raise HTTPException(404, "Document not found")
    return {"documents": rag.list_documents()}


@app.post("/api/chat")
def chat(body: ChatIn):
    def events():
        try:
            for event in rag.answer_stream(body.question, [t.model_dump() for t in body.history], body.source, body.mode):
                yield json.dumps(event) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": f"The model request failed ({type(e).__name__}). Try again in a moment."}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# The interface is built on a licensed template, so it is kept out of the public repo.
# Locally it lives in web/; on Render it is uploaded as a Secret File (/etc/secrets/index.html).
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
SECRET_INDEX = os.getenv("WEB_INDEX", "/etc/secrets/index.html")

if os.path.isfile(os.path.join(WEB_DIR, "index.html")):
    # registered last so /api routes win
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:
    @app.get("/", include_in_schema=False)
    def index():
        if not os.path.isfile(SECRET_INDEX):
            return HTMLResponse("<h1>NeuralDocs API is running</h1><p>The web interface is not installed. "
                                "See <a href='/api/docs'>/api/docs</a> for the API.</p>", status_code=503)
        return FileResponse(SECRET_INDEX, media_type="text/html")
