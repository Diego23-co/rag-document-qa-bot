# NeuralDocs — RAG Document Q&A Bot

> An AI-powered document intelligence system that lets you upload PDFs and have natural conversations about their contents — with zero hallucinations and full source citations.

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
│  Embed Question  │  ← sentence-transformers (local)
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
│  Stream Answer   │  ← LLaMA 3.1 via Groq API
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
| Dark UI | Professional dark-themed interface with NeuralDocs branding |

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM Inference | Groq — LLaMA 3.1 8B | Free tier, extremely fast |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free, no API needed |
| Vector Database | ChromaDB | Local, persistent, zero setup |
| PDF Parsing | pdfplumber | Reliable text extraction |
| UI Framework | Streamlit | Rapid Python web apps |
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

5. Ingest your documents

```bash
python ingest.py
```

6. Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` and start asking questions.

---

## Project Structure

```
neuraldocs/
├── app.py              # Streamlit web UI — main application
├── ingest.py           # PDF ingestion pipeline
├── chat.py             # Terminal-based chat interface
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this)
├── .gitignore          # Git ignore rules
├── docs/               # Drop your PDFs here
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
streamlit run app.py
```

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
MODEL = "llama-3.1-8b-instant"  # Groq model to use
```

---

## Roadmap

- [ ] Support for .docx and .txt files
- [ ] User authentication for multi-user deployments
- [ ] Chat history export
- [ ] Deployment to cloud (Render / Railway)
- [ ] Support for scanned PDFs via OCR

---

## Author

Khutso Mashapu
BSc Hons Computer Science — 2026

GitHub: https://github.com/Diego23-co
LinkedIn: https://linkedin.com/in/khutsomashapu

---

## License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

> Built as part of an AI/ML engineering learning journey. This project demonstrates core RAG architecture skills relevant to production AI systems.