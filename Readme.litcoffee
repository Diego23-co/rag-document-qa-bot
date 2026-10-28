# NeuralDocs — RAG Document Q&A Bot

> An AI-powered document intelligence system that lets you upload PDFs and have natural conversations about their contents — with zero hallucinations and full source citations.

**Live demo:** _link coming soon_ — the free server sleeps when idle, so the first visit can take up to a minute to wake up.

![NeuralDocs home screen with the document library and live ASCII jellyfish background](screenshots/home.png)

| Answers with clickable page citations | On a phone |
|---|---|
| ![An answer with numbered citations and the cited passage from page 2 opened](screenshots/chat.png) | ![The chat view on a 390 px wide phone screen](screenshots/phone.png) |

---

## Overview

NeuralDocs is a Retrieval-Augmented Generation (RAG) application built from scratch. Instead of relying solely on an LLM's training knowledge, it retrieves relevant context directly from your documents and grounds every answer in your actual content.

This means:
- Answers are always based on your documents
- Every response includes source citations
- Zero hallucinations — if it's not in the doc, the bot says so
- Works entirely for free with local embeddings and Groq's free API tier

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│  Embed Question  │  ← all-MiniLM-L6-v2 via ONNX (local)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChromaDB Search │  ← semantic similarity search
│  (Vector Store)  │     retrieves top 3 relevant chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build Prompt    │  ← inject retrieved chunks as context
│  + Send to LLM   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stream Answer   │  ← GPT-OSS via Groq API
│  + Show Sources  │
└─────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| Multi-PDF Ingestion | Upload and ingest multiple PDFs at once via drag and drop |
| Semantic Search | Finds the most relevant chunks using vector similarity |
| Source Citations | Every answer shows exactly which document it came from |
| Streaming Responses | Typewriter effect — answers appear word by word |
| Document Focus | Narrow the bot's search to a single document |
| Delete Documents | Remove documents from the knowledge base at any time |
| Input Locking | Input disables while streaming to prevent race conditions |
| Follow-up Suggestions | Bot suggests natural follow-up questions after each answer |
| Page Citations | Answers cite passages inline as [1], [2] — click one to read the exact excerpt and page |
| Answer Modes | Auto, Fast (GPT-OSS 20B) or Think (GPT-OSS 120B, reads more passages) |
| Stop & Regenerate | Stop an answer mid-stream (Esc) or ask for a fresh one |
| Command Palette | ⌘K / Ctrl K to search past questions, focus a document or run commands |
| Export | Save a conversation as Markdown |
| Living Background | ASCII jellyfish drawn live on a canvas react to typing, answers and the chosen mode (pause anytime; respects reduced motion) |

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM Inference | Groq — GPT-OSS 20B / 120B | Free tier, extremely fast |
| Embeddings | all-MiniLM-L6-v2 via ONNX Runtime (bundled with ChromaDB) | Local, free, no API and no PyTorch, so it runs in about 300 MB of RAM |
| Vector Database | ChromaDB | Local, persistent, zero setup |
| PDF Parsing | pdfplumber | Reliable text extraction |
| Web Server | FastAPI + Uvicorn | Streams answers to the browser as NDJSON |
| Interface | HTML, CSS and JavaScript (no build step) | Fast to load, no framework to install |
| Legacy UI | Streamlit (`app.py`) | Kept for Streamlit Cloud deployments |
| Environment | python-dotenv | Secure API key management |

---

## Getting Started

### Prerequisites

- Python 3.10+
- A free Groq API key from https://console.groq.com (no credit card required)

### Installation

1. Clone the repository

```bash
git clone https://github.com/Diego23-co/neuraldocs.git
cd neuraldocs
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up your API key

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

4. Add your PDFs

Create a `docs/` folder and drop your PDF files inside:

```
neuraldocs/
├── docs/
│   ├── your_document.pdf
│   └── another_document.pdf
```

The web server indexes these automatically when it starts, and visitors can't delete them, so the public demo always has something to ask about. PDFs that visitors upload are shared with everyone and can be deleted from the documents list.

5. Ingest your documents

```bash
python ingest.py
```

6. Launch the app

```bash
uvicorn server:app --reload
```

Open your browser at `http://localhost:8000` and start asking questions. You can also drag PDFs straight onto the page instead of running `ingest.py`.

---

## Project Structure

```
neuraldocs/
├── server.py           # FastAPI server — API + serves the web interface
├── rag.py              # RAG core: ingestion, retrieval, streamed answers
├── web/index.html      # The web interface (not in the repo, see Deployment)
├── app.py              # Legacy Streamlit UI (Streamlit Cloud)
├── ingest.py           # PDF ingestion from the command line
├── chat.py             # Terminal-based chat interface
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this)
├── .gitignore          # Git ignore rules
├── docs/               # Sample PDFs, indexed on server start and protected from deletion
└── chroma_db/          # Auto-generated vector database
```

---

## Usage

### Ingesting Documents

```bash
# Ingest all PDFs in the docs/ folder
python ingest.py

# Ingest a specific PDF
python ingest.py path/to/file.pdf

# See what is currently stored
python ingest.py --list
```

### Chat via Terminal

```bash
python chat.py
```

### Chat via Web UI

```bash
uvicorn server:app --reload       # http://localhost:8000
```

The old Streamlit interface still works with `streamlit run app.py`.

### How answers are delivered

`POST /api/chat` streams newline-delimited JSON, so the browser can show progress before the answer is finished:

```
{"type": "sources",   "sources": [{"n": 1, "source": "report.pdf", "page": 4, "snippet": "..."}]}
{"type": "meta",      "model": "openai/gpt-oss-20b", "mode": "fast"}
{"type": "token",     "text": "The report finds"}            ← repeated as the answer streams
{"type": "followups", "items": ["...", "..."]}
{"type": "done",      "ms": 1840, "first_token_ms": 420}
```

Only the last 6 messages are sent with each question, and short follow-ups ("what about the second one?") reuse the previous question for retrieval. Other endpoints: `GET /api/status`, `GET/POST /api/documents`, `DELETE /api/documents/{name}`. Interactive API docs live at `/api/docs`.

---

## How RAG Works

Traditional LLMs answer from their training data — which has a knowledge cutoff and can hallucinate. RAG fixes this by:

1. Ingestion — your documents are split into chunks, converted to vectors (embeddings), and stored in ChromaDB
2. Retrieval — when you ask a question, it is also converted to a vector and the most semantically similar chunks are retrieved
3. Augmentation — those chunks are injected into the LLM's prompt as context
4. Generation — the LLM answers based only on the retrieved context, not its training data

This architecture is used in production AI systems at companies like Notion, Intercom, and Klarna.

---

## Configuration

You can tweak these values at the top of `ingest.py` and `chat.py`:

```python
CHUNK_SIZE = 500       # Characters per chunk — larger = more context per chunk
CHUNK_OVERLAP = 50     # Overlap between chunks — helps preserve boundary context
TOP_K = 3              # Number of chunks retrieved per query
MODEL = "openai/gpt-oss-20b"  # Groq model to use
```

The web server reads its settings from the top of `rag.py` (`CHUNK_SIZE = 600`, `CHUNK_OVERLAP = 80`, `TOP_K` per mode, `HISTORY_TURNS = 6`). Override the models with the `FAST_MODEL` and `DEEP_MODEL` environment variables.

---

## Deployment

The server needs about 270 MB of RAM, so it fits a free web service on Render or Koyeb (512 MB).

**About the interface:** the web interface (`web/index.html`) is built on a licensed template that can't be published, so it is not in this repository. Without it, the server still runs and the API works (`/api/docs`). The live demo serves it from a private file:

1. On Render, create a **Web Service** from this repository.
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
2. Under **Environment**, add `GROQ_API_KEY`, then add a **Secret File** named `index.html` containing the interface. Render places it at `/etc/secrets/index.html`, which the server serves at `/`.
3. PDFs in `docs/` are indexed on every start. Free instances wipe their disk on restart, which also clears visitor uploads.

---

## Roadmap

- [ ] Support for .docx and .txt files
- [ ] User authentication for multi-user deployments
- [x] Chat history export
- [ ] Deployment to cloud (Render)
- [ ] Support for scanned PDFs via OCR

---

## Author

Khutso Mashapu
BSc Hons Computer Science — 2026

GitHub: https://github.com/Diego23-co
LinkedIn: https://linkedin.com/in/khutsomashapu

---

## Credits

- Interface layout and the ASCII jellyfish background are adapted from the **Kurage Dashboard** template by HorizonX, used under the owner's HorizonX membership license. The template files are not redistributed in this repository.
- Icons: [Lucide](https://lucide.dev) (ISC License). Fonts: Geist and Geist Mono via Google Fonts (SIL Open Font License).

---

## License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

> Built as part of an AI/ML engineering learning journey. This project demonstrates core RAG architecture skills relevant to production AI systems.