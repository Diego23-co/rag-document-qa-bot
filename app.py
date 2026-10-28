import streamlit as st
import chromadb
from embeddings import Embedder
from groq import Groq
from dotenv import load_dotenv
import pdfplumber
import os
import tempfile

load_dotenv()


# CONFIG

COLLECTION_NAME = "documents"
TOP_K = 3
MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# PAGE SETUP

st.set_page_config(
    page_title="NeuralDocs — RAG Q&A",
    page_icon="⚡",
    layout="wide"
)

# CUSTOM STYLING

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Force sidebar always open — hide collapse button ── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { min-width: 280px !important; transform: none !important; }

/* ── Kill the white box around the chat input ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
.stChatFloatingInputContainer,
.stChatFloatingInputContainer > div,
section[data-testid="stBottom"] {
    background: #060610 !important;
    background-color: #060610 !important;
    border-top: 1px solid rgba(99,31,255,0.2) !important;
    box-shadow: none !important;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #060610;
    font-family: 'DM Sans', sans-serif;
}

/* ── Animated mesh background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,31,255,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,210,130,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 60% 30%, rgba(255,60,120,0.07) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.stAppHeader { display: none; }
[data-testid="stToolbar"] { display: none; }
.stMainBlockContainer { padding-top: 0 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c0c1e !important;
    border-right: 1px solid rgba(99,31,255,0.2) !important;
}
[data-testid="stSidebar"] > div {
    padding: 2rem 1.4rem;
}

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #631fff, #00d282);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.sidebar-tagline {
    font-size: 0.72rem;
    color: #44445a;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 2rem;
}

.sidebar-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #44445a;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 1.5rem 0 0.8rem;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(99,31,255,0.05) !important;
    border: 1px dashed rgba(99,31,255,0.3) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,31,255,0.6) !important;
}
[data-testid="stFileUploader"] label { color: #888 !important; font-size: 0.82rem !important; }

/* Doc pill */
.doc-pill {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99,31,255,0.08);
    border: 1px solid rgba(99,31,255,0.2);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin-bottom: 0.4rem;
    font-size: 0.78rem;
    color: #c4b5fd;
    font-family: 'DM Sans', sans-serif;
    transition: background 0.2s;
}
.doc-pill:hover { background: rgba(99,31,255,0.15); }
.doc-pill-icon { font-size: 0.9rem; }
.doc-pill-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-pill-count {
    font-size: 0.65rem;
    color: #631fff;
    background: rgba(99,31,255,0.15);
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
}

/* Clear button */
[data-testid="stButton"] button {
    background: rgba(255,60,80,0.08) !important;
    color: #ff3c50 !important;
    border: 1px solid rgba(255,60,80,0.2) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    background: rgba(255,60,80,0.15) !important;
    border-color: rgba(255,60,80,0.4) !important;
}

/* ── Main content area ── */
.main-wrapper {
    max-width: 820px;
    margin: 0 auto;
    padding: 3rem 2rem 8rem;
    position: relative;
    z-index: 1;
}

/* Hero header */
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(99,31,255,0.12);
    border: 1px solid rgba(99,31,255,0.25);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.72rem;
    color: #a78bfa;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    color: #f0f0ff;
    margin-bottom: 0.6rem;
}

.hero-title span {
    background: linear-gradient(135deg, #631fff 0%, #00d282 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #555570;
    margin-bottom: 2.5rem;
    line-height: 1.6;
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 1rem;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
}
.stat-chip {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.stat-chip-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #00d282;
}
.stat-chip-label {
    font-size: 0.72rem;
    color: #44445a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Divider */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,31,255,0.3), transparent);
    margin: 2rem 0;
}

/* ── Chat messages ── */
.msg-wrap { margin-bottom: 1.2rem; animation: fadeUp 0.3s ease; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.msg-user {
    background: linear-gradient(135deg, rgba(99,31,255,0.12), rgba(99,31,255,0.05));
    border: 1px solid rgba(99,31,255,0.2);
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.3rem;
    max-width: 85%;
    margin-left: auto;
}

.msg-bot {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px 16px 16px 16px;
    padding: 1rem 1.3rem;
    max-width: 95%;
}

.msg-bot:hover {
    border-color: rgba(0,210,130,0.2);
    background: rgba(0,210,130,0.02);
    transition: all 0.2s;
}

.role-tag {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.role-tag-user { color: #a78bfa; }
.role-tag-bot  { color: #00d282; }

.msg-text {
    font-size: 0.92rem;
    color: #d0d0e8;
    line-height: 1.75;
}

.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.8rem;
}
.source-chip {
    font-size: 0.68rem;
    color: #631fff;
    background: rgba(99,31,255,0.1);
    border: 1px solid rgba(99,31,255,0.2);
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-family: 'DM Sans', sans-serif;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #2a2a3e;
}
.empty-state-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-state-text { font-family: 'Syne', sans-serif; font-size: 1rem; color: #3a3a5e; }

/* Fix typed text color — input has light bg so text must be dark */
[data-testid="stChatInputTextArea"] textarea,
[data-testid="stChatInput"] textarea,
.stChatInput textarea,
div[class*="chatInput"] textarea,
textarea {
    color: #111111 !important;
    caret-color: #631fff !important;
}
textarea::placeholder {
    color: #888899 !important;
    opacity: 1 !important;
}

/* Chat input override */
[data-testid="stChatInput"] {
    background: rgba(12,12,30,0.98) !important;
    border-top: 1px solid rgba(99,31,255,0.2) !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(99,31,255,0.3) !important;
    border-radius: 12px !important;
    color: #111111 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    caret-color: #631fff !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #7070a0 !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99,31,255,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,31,255,0.1) !important;
    background: rgba(255,255,255,0.09) !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #631fff, #00d282) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Spinner */
[data-testid="stSpinner"] { color: #631fff !important; }

/* Success / warning toasts */
.stAlert {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
}
</style>
""", unsafe_allow_html=True)


# CACHED RESOURCES

@st.cache_resource
def load_embedder():
    return Embedder()

@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(COLLECTION_NAME)

@st.cache_resource
def load_groq():
    # Try Streamlit Cloud secrets first (for deployment)
    # Fall back to .env file for local development
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to your .env file or Streamlit Cloud secrets.")
        st.stop()
    return Groq(api_key=api_key)

embedder = load_embedder()
collection = load_db()
groq_client = load_groq()

# INGESTION HELPERS

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def chunk_text(text):
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def ingest_pdf_bytes(file_bytes, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
        if not text.strip():
            return 0
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks).tolist()
        existing = collection.get(where={"source": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))]
        )
        return len(chunks)
    finally:
        os.unlink(tmp_path)


# RAG QUERY

# Rotating thinking messages — makes the bot feel alive while it processes
THINKING_MESSAGES = [
    "🔍 Scanning your documents...",
    "🧠 Thinking this through...",
    "📖 Reading through your files...",
    "⚡ Finding the best answer...",
    "💡 Connecting the dots...",
    "🎯 Zeroing in on the answer...",
]

def ask(question, history):
    qe = embedder.encode([question]).tolist()

    # Filter to a specific document if one is selected, otherwise search all
    selected = st.session_state.get("selected_doc", "All Documents")
    where_filter = {"source": selected} if selected != "All Documents" else None

    results = collection.query(
        query_embeddings=qe,
        n_results=TOP_K,
        where=where_filter,
        include=["documents", "metadatas"]
    )
    chunks = results["documents"][0]
    sources = list(dict.fromkeys(m["source"] for m in results["metadatas"][0]))
    context = "\n\n---\n\n".join(chunks)

    # Humanised personality prompt — warm, direct, and conversational
    system_prompt = f"""You are NeuralDocs, a smart and friendly AI assistant with a warm, direct personality — like a knowledgeable friend who has read all the documents.

CRITICAL RULES — follow these strictly:
- ALWAYS synthesise and answer in your own words — NEVER copy or paste raw text from the context
- Use the document context ONLY as a source of facts to answer with
- If the answer isn't in the documents, say "Hmm, I couldn't find that one — try rephrasing or switching documents!"
- If multiple chunks are retrieved, combine them into one coherent answer
- Use natural conversational language — no stiff corporate phrasing
- Use emojis sparingly (1-2 max per response) to keep things warm
- Be concise but never cold
- After your answer, suggest 1 natural follow-up question prefixed with: "💬 You might also want to ask:"

DOCUMENT CONTEXT (use as reference only — do NOT copy this text directly):
{context}"""

    # Stream the response token by token for the typewriter effect
    stream = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}] + history,
        temperature=0.5,    # slightly higher = more natural, human-sounding responses
        max_tokens=1024,
        stream=True         # enables word-by-word streaming
    )
    return stream, sources

# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

# Tracks whether the bot is currently streaming a response
# Used to disable the input field while streaming
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False


if "ingested_files" not in st.session_state:
    all_data = collection.get()
    if all_data["metadatas"]:
        sources = {}
        for m in all_data["metadatas"]:
            sources[m["source"]] = sources.get(m["source"], 0) + 1
        st.session_state.ingested_files = sources   # {filename: chunk_count}
    else:
        st.session_state.ingested_files = {}

# SIDEBAR

with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ NeuralDocs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">RAG-Powered Document Intelligence</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">📁 Upload PDFs</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.ingested_files:
                with st.spinner(f"Processing {uf.name}..."):
                    n = ingest_pdf_bytes(uf.read(), uf.name)
                    if n > 0:
                        st.session_state.ingested_files[uf.name] = n
                        st.success(f"✅ {uf.name}")
                    else:
                        st.warning(f"⚠️ {uf.name} — no text found")

    if st.session_state.ingested_files:
        st.markdown('<div class="sidebar-section-title">📚 Knowledge Base</div>', unsafe_allow_html=True)

        # Loop through each document and show it with a delete button
        for fname, count in list(st.session_state.ingested_files.items()):
            short = fname if len(fname) < 22 else fname[:19] + "..."

            # Two columns: doc pill on left, delete button on right
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(f"""
                    <div class="doc-pill">
                        <span class="doc-pill-icon">📄</span>
                        <span class="doc-pill-name">{short}</span>
                        <span class="doc-pill-count">{count} chunks</span>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                # Each button needs a unique key to avoid Streamlit conflicts
                if st.button("✕", key=f"delete_{fname}", help=f"Remove {fname}"):
                    # Delete all chunks for this document from ChromaDB
                    existing = collection.get(where={"source": fname})
                    if existing["ids"]:
                        collection.delete(ids=existing["ids"])

                    # Remove from session state tracking
                    del st.session_state.ingested_files[fname]

                    # If this was the focused document, reset focus to All Documents
                    if st.session_state.get("selected_doc") == fname:
                        st.session_state.selected_doc = "All Documents"

                    st.rerun()

        # Document focus selector — lets user narrow search to one PDF
        st.markdown('<div class="sidebar-section-title">🎯 Focus</div>', unsafe_allow_html=True)
        selected_doc = st.selectbox(
            "Focus on document:",
            ["All Documents"] + list(st.session_state.ingested_files.keys()),
            label_visibility="collapsed"
        )
        st.session_state.selected_doc = selected_doc   # save to session state

    st.markdown('<div class="sidebar-section-title">⚙️ Actions</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# MAIN CONTENT

st.markdown('<div class="main-wrapper">', unsafe_allow_html=True)

# Hero
st.markdown("""
    <div class="hero-badge">⚡ Powered by GPT-OSS + RAG</div>
    <div class="hero-title">Ask anything about<br><span>your documents.</span></div>
    <div class="hero-subtitle">Upload PDFs and get precise, source-cited answers instantly.<br>
    No hallucinations — every answer is grounded in your documents.</div>
""", unsafe_allow_html=True)

# Stats bar
doc_count = len(st.session_state.ingested_files)
chunk_count = sum(st.session_state.ingested_files.values())
msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])

st.markdown(f"""
    <div class="stats-bar">
        <div class="stat-chip">
            <span class="stat-chip-value">{doc_count}</span>
            <span class="stat-chip-label">Documents</span>
        </div>
        <div class="stat-chip">
            <span class="stat-chip-value">{chunk_count}</span>
            <span class="stat-chip-label">Chunks indexed</span>
        </div>
        <div class="stat-chip">
            <span class="stat-chip-value">{msg_count}</span>
            <span class="stat-chip-label">Questions asked</span>
        </div>
    </div>
    <div class="section-divider"></div>
""", unsafe_allow_html=True)

# Chat history
if not st.session_state.messages:
    st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <div class="empty-state-text">Upload a PDF and ask your first question</div>
        </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
                <div class="msg-wrap">
                    <div class="msg-user">
                        <div class="role-tag role-tag-user">● You</div>
                        <div class="msg-text">{msg["content"]}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            sources_html = ""
            if msg.get("sources"):
                chips = "".join(f'<span class="source-chip">📄 {s}</span>' for s in msg["sources"])
                sources_html = f'<div class="sources-row">{chips}</div>'
            st.markdown(f"""
                <div class="msg-wrap">
                    <div class="msg-bot">
                        <div class="role-tag role-tag-bot">⚡ NeuralDocs</div>
                        <div class="msg-text">{msg["content"]}</div>
                        {sources_html}
                    </div>
                </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# CHAT INPUT

import random

# Disable the input field while the bot is streaming a response
# This prevents new questions being submitted mid-stream causing lag/overlap
user_input = st.chat_input(
    "Ask a question about your documents..." if not st.session_state.is_streaming else "⏳ Wait for the response to finish...",
    disabled=st.session_state.is_streaming
)

if user_input:
    if not st.session_state.ingested_files:
        st.warning("⚠️ Please upload at least one PDF first using the sidebar.")
    else:
        # Lock the input immediately so user can't submit another question
        st.session_state.is_streaming = True

        # Show user message immediately
        st.markdown(f"""
            <div class="msg-wrap">
                <div class="msg-user">
                    <div class="role-tag role-tag-user">● You</div>
                    <div class="msg-text">{user_input}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        # Pick a random thinking message so it feels different every time
        thinking_msg = random.choice(THINKING_MESSAGES)

        with st.spinner(thinking_msg):
            stream, sources = ask(user_input, history)

        # Render the bot bubble header
        chips = "".join(f'<span class="source-chip">📄 {s}</span>' for s in sources)
        st.markdown(f"""
            <div class="msg-wrap">
                <div class="msg-bot">
                    <div class="role-tag role-tag-bot">⚡ NeuralDocs</div>
            </div>
        """, unsafe_allow_html=True)

        # Stream the answer word by word — typewriter effect
        answer_container = st.empty()
        full_answer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_answer += delta
            answer_container.markdown(f"""
                <div class="msg-text" style="padding: 0 1.3rem 0.5rem;">{full_answer}▌</div>
            """, unsafe_allow_html=True)

        # Final render — remove cursor, show sources
        answer_container.markdown(f"""
            <div class="msg-text" style="padding: 0 1.3rem 1rem;">{full_answer}
                <div class="sources-row" style="margin-top:0.8rem;">{chips}</div>
            </div>
        """, unsafe_allow_html=True)

        # Save messages to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": full_answer, "sources": sources})

        # Unlock the input now that streaming is complete
        st.session_state.is_streaming = False
        st.rerun()