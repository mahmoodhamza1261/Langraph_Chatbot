"""
NODES/RETRIEVE.PY — Searches the vector store for relevant document chunks.

WHY THIS NODE EXISTS:
When the user asks something that should be answered from our documents
(classified as "rag"), we need to FIND the relevant parts first.
We can't send 100 pages of documents to the LLM — that would be too slow
and expensive. Instead, we search for just the relevant chunks.

HOW VECTOR SEARCH WORKS:
1. The user's question is converted to a vector (a list of numbers)
   Example: "How does LangGraph handle state?" → [0.12, -0.34, 0.89, ...]
2. We search the vector database for stored chunks with similar vectors
3. "Similar" = vectors that are mathematically close (cosine similarity)
4. We return the top-k closest chunks (k=4 means 4 chunks)

WHY THIS IS BETTER THAN KEYWORD SEARCH:
Keyword search: "LangGraph state" only finds docs containing those exact words
Vector search: "How does LangGraph handle state?" also finds docs about
  "TypedDict", "add_messages", "reducer" — because they're semantically related

IMPORTANT: This node doesn't call the LLM at all!
It only searches the vector database. Fast and cheap.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from state import GraphState

# Initialize embedding model and vector store once at module level
# This avoids reloading the 90MB model on every function call
print("[SETUP] Loading embedding model for retriever...")
_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

_vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=_embeddings
)

# Create a retriever from the vector store
# k=4 means: return the 4 most relevant chunks
# search_type="similarity" uses cosine similarity (default)
_retriever = _vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


def retrieve(state: GraphState) -> dict:
    """
    NODE: retrieve

    Searches the vector store with the current query.
    Uses state["query"] — which might be the original question,
    or a REWRITTEN question if we're on retry.

    Input from state:
        - state["query"]: the question to search with
          (could be original or rewritten by rewrite_query node)

    Output to state:
        - documents: list of relevant text chunks (strings)

    Note: This node is called TWICE in some cases:
    - First time: with the original question
    - After rewrite_query: with the improved question
    This is possible because of the loop edge: rewrite_query → retrieve
    """
    query = state["query"]
    retry = state.get("retry_count", 0)

    print(f"\n[NODE: retrieve]")
    print(f"  Query: {query}")
    print(f"  Attempt: {retry + 1}")

    # Search the vector store
    # Returns a list of Document objects with .page_content and .metadata
    docs = _retriever.invoke(query)

    # Extract just the text content (not the metadata objects)
    doc_texts = [doc.page_content.strip() for doc in docs]

    print(f"  Found {len(doc_texts)} chunks")
    for i, text in enumerate(doc_texts, 1):
        print(f"  Chunk {i}: {text[:80]}...")

    return {"documents": doc_texts}
