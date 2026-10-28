import pdfplumber       # extracts text from PDF files
import chromadb         # local vector database to store embeddings
from embeddings import Embedder  # local all-MiniLM-L6-v2 embeddings via ONNX (free, no API, no PyTorch)
import os               # for file and folder operations
import sys              # for reading command line arguments

# CONFIG: tweak these values to change behaviour

CHUNK_SIZE = 500        # how many characters per chunk (bigger = more context, less precise)
CHUNK_OVERLAP = 50      # how many characters overlap between chunks (helps preserve context at boundaries)
COLLECTION_NAME = "documents"   # name of the ChromaDB collection (like a table name)
DOCS_FOLDER = "./docs"          # default folder to look for PDFs when no argument is passed


# SETUP: load model and connect to database (runs once)

# Load the local sentence embedding model
# This converts text → numerical vectors so we can do similarity search
# "all-MiniLM-L6-v2" is small, fast, and free, and runs entirely on your machine
print("Loading embedding model...")
embedder = Embedder()

# Connect to ChromaDB, a local vector database stored in the ./chroma_db folder
# PersistentClient means data survives after the script closes (saved to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Get or create a collection (like a table) called "documents"
# All PDF chunks from all documents go into this one collection
collection = client.get_or_create_collection(COLLECTION_NAME)


# STEP 1, EXTRACT: pull raw text out of a PDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF and extracts all text from every page.
    Returns a single string with all the text joined together.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:                       # skip pages with no text (e.g. blank pages)
                text += page_text + "\n"        # add newline between pages
    return text


# STEP 2, CHUNK: split text into smaller overlapping pieces

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits a long text into smaller chunks with slight overlap.

    Why chunk? LLMs have a context limit, so we can't send an entire document.
    Why overlap? So sentences at chunk boundaries don't lose their context.

    Example with chunk_size=10, overlap=3:
    "Hello World Python Code" → ["Hello Worl", "orld Pytho", "thon Code"]
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():               # skip chunks that are only whitespace
            chunks.append(chunk)
        start += chunk_size - overlap   # move forward, but backtrack by overlap amount
    return chunks


# CORE FUNCTION: ingest a single PDF file

def ingest_pdf(pdf_path: str):
    """
    Full ingestion pipeline for one PDF:
    Extract text → Chunk it → Embed each chunk → Store in ChromaDB
    """
    filename = os.path.basename(pdf_path)   # get just the filename e.g. "resume.pdf"
    print(f"\n  📄 {filename}")

    # --- Step 1: Extract ---
    text = extract_text_from_pdf(pdf_path)

    # Guard: skip if no text was found (common with scanned/image-based PDFs)
    if not text.strip():
        print(f"     ⚠️  No text found, skipping (may be a scanned image PDF)")
        return

    print(f"     Extracted {len(text)} characters")

    # --- Step 2: Chunk ---
    chunks = chunk_text(text)
    print(f"     Created {len(chunks)} chunks")

    # --- Step 3: Embed ---
    # Convert each chunk from text → vector (list of numbers)
    # The embedder returns a numpy array, .tolist() converts it for ChromaDB
    embeddings = embedder.encode(chunks).tolist()

    # --- Deduplication: remove old version of this file if it was ingested before ---
    # This prevents duplicate chunks if you re-ingest the same PDF
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        print(f"     ♻️  Replaced previous version")

    # --- Step 4: Store ---
    # Each chunk needs a unique ID, so we use filename + chunk index
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,           # the raw text of each chunk
        embeddings=embeddings,      # the vector representation of each chunk
        ids=ids,                    # unique identifier for each chunk
        metadatas=[                 # extra info stored alongside each chunk
            {"source": filename, "chunk_index": i}
            for i in range(len(chunks))
        ]
    )
    print(f"     ✅ Stored {len(chunks)} chunks")


# BATCH FUNCTION: ingest all PDFs in a folder

def ingest_folder(folder_path: str):
    """
    Finds all PDF files in a folder and ingests them one by one.
    Creates the folder if it doesn't exist yet.
    """
    # Create the docs folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created '{folder_path}' folder.")
        print(f"   Drop your PDF files in there and run this script again.")
        return

    # Find all PDF files in the folder (case-insensitive .pdf check)
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️  No PDF files found in '{folder_path}'.")
        print(f"   Drop your PDF files in the '{folder_path}' folder and try again.")
        return

    print(f"\n🗂️  Found {len(pdf_files)} PDF(s) in '{folder_path}':")

    # Loop through every PDF and ingest it
    success = 0
    for pdf_file in pdf_files:
        try:
            ingest_pdf(os.path.join(folder_path, pdf_file))    # build full path e.g. ./docs/resume.pdf
            success += 1
        except Exception as e:
            print(f"     ❌ Error ingesting {pdf_file}: {e}")  # don't crash, just skip bad files

    # Print a final summary
    total_chunks = collection.count()   # total chunks across ALL documents in ChromaDB
    print(f"\n{'='*45}")
    print(f"✅ Ingested {success}/{len(pdf_files)} PDFs successfully")
    print(f"📦 Total chunks stored in ChromaDB: {total_chunks}")
    print(f"{'='*45}")
    print(f"\nYou can now run: python chat.py")


# UTILITY: list all documents currently stored in ChromaDB

def list_ingested():
    """
    Shows all documents currently stored in ChromaDB and how many chunks each has.
    Run with: python ingest.py --list
    """
    all_data = collection.get()     # fetch everything from ChromaDB

    if not all_data["ids"]:
        print("No documents ingested yet.")
        return

    # Extract unique document names from metadata
    sources = set(m["source"] for m in all_data["metadatas"])

    print(f"\n📚 Currently ingested documents ({len(sources)}):")
    for source in sorted(sources):
        # Count how many chunks belong to this document
        count = sum(1 for m in all_data["metadatas"] if m["source"] == source)
        print(f"   • {source} ({count} chunks)")


# ENTRY POINT: handles command line arguments

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "--list":
            # python ingest.py --list → show what's stored
            list_ingested()

        elif sys.argv[1].endswith(".pdf"):
            # python ingest.py resume.pdf → ingest a single file
            ingest_pdf(sys.argv[1])
            print(f"\nRun: python chat.py")

        else:
            # python ingest.py ./my_folder → ingest all PDFs in a custom folder
            ingest_folder(sys.argv[1])
    else:
        # python ingest.py → default: ingest all PDFs in ./docs folder
        ingest_folder(DOCS_FOLDER)
