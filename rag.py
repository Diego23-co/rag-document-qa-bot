"""
NeuralDocs RAG core, shared by the web server (server.py).

Ingest:   PDF bytes -> text per page -> word-aware chunks -> embeddings -> ChromaDB
Answer:   question -> retrieve top chunks -> grounded prompt -> streamed Groq answer
"""

import io
import os
import re
import threading
import time
from collections import Counter

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # no ChromaDB usage pings

import chromadb
import pdfplumber
from dotenv import load_dotenv
from groq import Groq, NotFoundError

from embeddings import Embedder

load_dotenv()

# CONFIG

COLLECTION_NAME = "documents"       # same collection as ingest.py and app.py
DB_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CHUNK_SIZE = 600                    # characters per chunk
CHUNK_OVERLAP = 80                  # characters shared between neighbouring chunks
HISTORY_TURNS = 6                   # past messages sent to the LLM (keeps requests small)
MODELS = {
    "fast": os.getenv("FAST_MODEL", "openai/gpt-oss-20b"),
    "think": os.getenv("DEEP_MODEL", "openai/gpt-oss-120b"),
}
REASONING = {"fast": "low", "think": "medium"}   # gpt-oss only: how long it thinks before answering
TOP_K = {"fast": 4, "think": 6}
FOLLOWUP_MARK = "<<<FOLLOWUPS>>>"   # the model writes follow-up questions after this line

# SETUP — loaded once when the server starts

embedder = Embedder()                # all-MiniLM-L6-v2 via ONNX, same vectors as ingest.py/app.py
collection = chromadb.PersistentClient(path=DB_PATH).get_or_create_collection(COLLECTION_NAME)
write_lock = threading.Lock()       # one ingest/delete at a time


def _groq_key():
    return os.getenv("GROQ_API_KEY")


groq_client = Groq(api_key=_groq_key()) if _groq_key() else None


# INGESTION

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking on whitespace instead of mid-word."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            if cut > start:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        # start the next chunk on a word boundary inside the overlap window
        nxt = text.find(" ", end - overlap, end)
        start = nxt + 1 if nxt != -1 else end
    return chunks


def ingest_pdf(data: bytes, filename: str) -> int:
    """Index one PDF, replacing any earlier version. Returns the number of chunks stored."""
    chunks, metas = [], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for chunk in chunk_text(page.extract_text() or ""):
                metas.append({"source": filename, "page": page_no, "chunk_index": len(chunks)})
                chunks.append(chunk)
    if not chunks:
        return 0

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    with write_lock:
        delete_document(filename, locked=True)
        for i in range(0, len(chunks), 1000):   # stay under Chroma's batch limit
            collection.add(documents=chunks[i:i + 1000], embeddings=embeddings[i:i + 1000],
                           ids=ids[i:i + 1000], metadatas=metas[i:i + 1000])
    return len(chunks)


def delete_document(filename: str, locked: bool = False) -> int:
    def run():
        ids = collection.get(where={"source": filename})["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    if locked:
        return run()
    with write_lock:
        return run()


def list_documents() -> list[dict]:
    counts = Counter(m["source"] for m in collection.get(include=["metadatas"])["metadatas"])
    samples = sample_names()
    return [{"name": name, "chunks": n, "sample": name in samples}
            for name, n in sorted(counts.items(), key=lambda kv: kv[0].lower())]


# DEMO SAMPLES — PDFs in ./docs are indexed on startup and cannot be deleted from the web UI

SAMPLES_DIR = os.getenv("SAMPLES_DIR", "./docs")


def sample_names() -> set[str]:
    if not os.path.isdir(SAMPLES_DIR):
        return set()
    return {f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".pdf")}


def load_samples() -> None:
    """Index any sample PDF that is missing (free hosts wipe the disk on restart)."""
    indexed = {d["name"] for d in list_documents()}
    for name in sorted(sample_names() - indexed):
        try:
            with open(os.path.join(SAMPLES_DIR, name), "rb") as f:
                n = ingest_pdf(f.read(), name)
            print(f"Indexed sample {name}: {n} chunks", flush=True)
        except Exception as e:
            print(f"Could not index sample {name}: {e}", flush=True)


# RETRIEVAL

def retrieve(query: str, source: str | None, k: int) -> list[dict]:
    total = collection.count()
    if not total:
        return []
    results = collection.query(
        query_embeddings=embedder.encode([query]).tolist(),
        n_results=min(k, total),
        where={"source": source} if source else None,
        include=["documents", "metadatas"],
    )
    return [
        {"n": i + 1, "source": meta["source"], "page": meta.get("page"), "text": text}
        for i, (text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]))
    ]


def search_query(question: str, history: list[dict]) -> str:
    """Short follow-ups ("what about the second one?") borrow the previous question for retrieval."""
    if len(question.split()) >= 8:
        return question
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    return f"{previous}\n{question}" if previous else question


def pick_mode(mode: str, question: str) -> str:
    if mode in MODELS:
        return mode
    deep = len(question) > 160 or re.search(r"\b(compare|contrast|why|analy[sz]e|explain|evaluate|summari[sz]e)\b", question, re.I)
    return "think" if deep else "fast"


# GENERATION

SYSTEM_PROMPT = """You are NeuralDocs, a precise and friendly assistant. Answer using only the numbered excerpts from the user's documents below.

How to answer:
- Put the direct answer in the first sentence, then only the detail that helps.
- Be brief: usually 2-5 sentences, or a short bulleted list when listing things. Use **bold** for key terms. No headings, no tables.
- Cite excerpts inline with their numbers in square brackets, like [1] or [2][3], right after the claim they support.
- If the excerpts do not contain the answer, say so in one sentence and suggest a better question. Never invent facts.
- Refer to "your documents", never to "excerpts" or "context".
- Finish with a line containing only {mark} followed by two short follow-up questions the documents can answer, one per line.

EXCERPTS:
{excerpts}"""


def _excerpts(passages: list[dict]) -> str:
    def label(p):
        return f"{p['source']}, p. {p['page']}" if p.get("page") else p["source"]
    return "\n\n".join(f"[{p['n']}] ({label(p)})\n{p['text']}" for p in passages)


def _trim_history(history: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"][:1500]} for m in history[-HISTORY_TURNS:]]


def _split_stream(deltas):
    """Yield answer text, holding back anything that could be the start of the follow-up marker.
    Returns the text after the marker (the follow-up questions) via StopIteration.value."""
    buf, tail, in_tail = "", "", False
    for d in deltas:
        if in_tail:
            tail += d
            continue
        buf += d
        i = buf.find(FOLLOWUP_MARK)
        if i >= 0:
            if buf[:i]:
                yield buf[:i]
            tail, buf, in_tail = buf[i + len(FOLLOWUP_MARK):], "", True
            continue
        keep = next((k for k in range(min(len(buf), len(FOLLOWUP_MARK) - 1), 0, -1)
                     if FOLLOWUP_MARK.startswith(buf[-k:])), 0)
        if len(buf) > keep:
            yield buf[:len(buf) - keep]
            buf = buf[len(buf) - keep:]
    if buf:
        yield buf
    return tail


def _parse_followups(tail: str) -> list[str]:
    lines = (re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in tail.splitlines())
    return [line for line in lines if len(line) > 3][:3]


def answer_stream(question: str, history: list[dict], source: str | None, mode: str):
    """Yield NDJSON-ready events: meta, sources, token..., followups, done (or error)."""
    started = time.perf_counter()
    if groq_client is None:
        yield {"type": "error", "message": "GROQ_API_KEY is missing. Add it to the .env file and restart the server."}
        return

    mode = pick_mode(mode, question)
    history = _trim_history(history)
    passages = retrieve(search_query(question, history), source, TOP_K[mode])
    if not passages:
        yield {"type": "error", "message": "There are no documents to search yet. Upload a PDF first."}
        return

    yield {"type": "sources", "sources": [
        {"n": p["n"], "source": p["source"], "page": p["page"], "snippet": p["text"][:420]} for p in passages
    ]}

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(mark=FOLLOWUP_MARK, excerpts=_excerpts(passages))}]
    messages += history + [{"role": "user", "content": question}]

    # try the chosen model first, then the other one (models get retired on Groq)
    stream = None
    for model in dict.fromkeys([MODELS[mode], MODELS["fast"], MODELS["think"]]):
        extra = {"reasoning_effort": REASONING[mode]} if model.startswith("openai/gpt-oss") else {}
        try:
            stream = groq_client.chat.completions.create(
                model=model, messages=messages, temperature=0.3, max_tokens=2000, stream=True, **extra)
            break
        except NotFoundError:
            continue
    if stream is None:
        yield {"type": "error", "message": f"Groq no longer offers {MODELS['fast']} or {MODELS['think']}. "
                                           "Set FAST_MODEL and DEEP_MODEL in .env to models listed at console.groq.com."}
        return
    yield {"type": "meta", "model": model, "mode": mode}

    deltas = (c.choices[0].delta.content or "" for c in stream if c.choices)
    splitter = _split_stream(deltas)
    first = None
    while True:
        try:
            text = next(splitter)
        except StopIteration as stop:
            followups = _parse_followups(stop.value or "")
            break
        if first is None:
            first = time.perf_counter()
        # gpt-oss sometimes cites with full-width brackets: 【1】 -> [1]
        yield {"type": "token", "text": text.replace("【", "[").replace("】", "]")}

    if followups:
        yield {"type": "followups", "items": followups}
    yield {"type": "done", "ms": round((time.perf_counter() - started) * 1000),
           "first_token_ms": round(((first or time.perf_counter()) - started) * 1000)}
