import chromadb                    # local vector database to search stored chunks
from embeddings import Embedder    # same embedding model used in ingest.py
from groq import Groq              # Groq API client for LLM inference (free tier)
from dotenv import load_dotenv     # loads environment variables from .env file
import os                          # for reading environment variables

# Load API key from .env file
load_dotenv()

# CONFIG
COLLECTION_NAME = "documents"       # must match the collection name used in ingest.py
TOP_K = 3                           # how many chunks to retrieve per query
                                    # higher = more context, but also more tokens sent to LLM
MODEL = "openai/gpt-oss-20b"        # free Groq model — fast and capable enough for Q&A

# SETUP — load model and connect to database (runs once)

# Load the same embedding model used during ingestion
# IMPORTANT: must be the same model — different models produce incompatible vectors
print("Loading embedding model...")
embedder = Embedder()

# Connect to the same ChromaDB database created by ingest.py
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(COLLECTION_NAME)

# Connect to Groq using your API key from the .env file
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# CONVERSATION HISTORY

# Stores the full conversation so the LLM remembers previous messages
# Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
conversation_history = []


# STEP 1 — RETRIEVE: find the most relevant chunks for a query

def retrieve_context(query: str, top_k: int = TOP_K) -> tuple[str, list[str]]:
    """
    Embeds the user's question and searches ChromaDB for the most
    semantically similar chunks.

    Returns:
      - context: all retrieved chunks joined into one string (sent to LLM)
      - sources: list of source filenames (shown to user)
    """
    # Embed the query using the same model used during ingestion
    # This converts the question into a vector so we can compare it to stored chunks
    query_embedding = embedder.encode([query]).tolist()

    # Search ChromaDB for the top_k most similar chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas"]  # return both text and metadata
    )

    # Extract the text chunks from results
    chunks = results["documents"][0]

    # Extract source filenames from metadata (used to show the user where the answer came from)
    sources = [m["source"] for m in results["metadatas"][0]]

    # Join chunks into a single context block separated by dividers
    context = "\n\n---\n\n".join(chunks)

    return context, sources


# STEP 2 — ANSWER: send context + question to the LLM

def ask(user_question: str) -> tuple[str, list[str]]:
    """
    Full RAG query pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Build a grounded system prompt with those chunks
    3. Send to Groq LLM and return the answer + sources
    """

    # Step 1: Retrieve relevant document chunks
    context, sources = retrieve_context(user_question)

    # Step 2: Build a system prompt that grounds the LLM in the retrieved context
    # The system prompt tells the LLM how to behave and gives it the document context
    system_prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document context.

If the answer is not found in the context, say: "I couldn't find that in the document."
Do not make up information. Be concise and clear.

DOCUMENT CONTEXT:
{context}
"""

    # Step 3: Add the user's question to the conversation history
    conversation_history.append({"role": "user", "content": user_question})

    # Step 4: Send the full conversation to Groq
    # We prepend the system prompt to every request (it's not stored in history)
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}] + conversation_history,
        temperature=0.2,    # low temperature = more factual, less creative/random
        max_tokens=1024     # max length of the response
    )

    # Step 5: Extract the response text
    answer = response.choices[0].message.content

    # Step 6: Add the assistant's response to history for future context
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, sources


# MAIN CHAT LOOP

def main():
    print("\n🤖 RAG Document Q&A Bot")
    print("=" * 40)
    print("Make sure you've run: python ingest.py")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        # Get input from the user
        user_input = input("You: ").strip()

        # Skip empty input (user just pressed Enter)
        if not user_input:
            continue

        # Exit condition
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Get answer and sources from the RAG pipeline
        print("\nBot: ", end="", flush=True)
        answer, sources = ask(user_input)
        print(answer)

        # Show which documents the answer was pulled from
        # Deduplicate sources in case multiple chunks came from the same file
        unique_sources = list(dict.fromkeys(sources))
        print(f"\n📎 Sources: {', '.join(unique_sources)}\n")


if __name__ == "__main__":
    main()
