"""
VECTORSTORE_SETUP.PY — One-time setup to build your chatbot's knowledge base.

WHY DO WE NEED THIS?
LLMs only know what they were trained on. If you want your chatbot to answer
questions about YOUR documents (company docs, PDFs, notes, etc.) you need RAG.

THE PROCESS:
1. Write raw text documents
2. Split them into small chunks (LLMs have context limits)
3. Convert each chunk into a vector (a list of numbers that captures meaning)
4. Store all vectors in ChromaDB (a vector database)
5. Later, when a user asks a question, we convert it to a vector too,
   and find the chunks whose vectors are CLOSEST — i.e., most similar in meaning

WHY CHUNKS?
Imagine a 500-page book. You can't send all 500 pages to the LLM every time.
Instead, you split it into 200-word chunks, store them all, and when a question
comes in, you only retrieve the 4-5 most relevant chunks.

WHY EMBEDDINGS?
"What is gravity?" and "How does gravity work?" look different as text.
But as vectors (embeddings), they're very close to each other numerically
because they mean almost the same thing. This is semantic search.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# ─── SAMPLE KNOWLEDGE BASE ───────────────────────────────────────────────────
# In a real app, you'd load PDFs, websites, or databases here.
# For this demo, we write the content directly.
DOCUMENTS = [
    Document(
        page_content="""
        LangGraph is a library built on top of LangChain for creating stateful,
        multi-actor applications with LLMs. It extends LangChain Expression Language
        with the ability to coordinate multiple chains (or actors) across multiple
        steps of computation in a cyclic manner.

        The key concepts in LangGraph are:
        - StateGraph: The main graph class that holds nodes and edges
        - Nodes: Python functions that receive state and return updates
        - Edges: Connections between nodes (can be conditional)
        - State: A TypedDict that flows through every node
        - Reducers: Functions that determine how state updates are merged

        LangGraph is particularly useful for building agents, multi-step RAG pipelines,
        and any application where you need cycles or loops in your LLM workflow.
        """,
        metadata={"source": "langgraph_docs", "topic": "langgraph"}
    ),
    Document(
        page_content="""
        RAG (Retrieval-Augmented Generation) is a technique that combines a retrieval
        system with a language model. Instead of relying solely on the LLM's training
        data, RAG fetches relevant information from a knowledge base at query time.

        The RAG pipeline works as follows:
        1. Indexing: Documents are split into chunks, embedded, and stored in a vector DB
        2. Retrieval: User query is embedded and similar chunks are retrieved
        3. Generation: Retrieved chunks + user query are sent to LLM to generate answer

        Benefits of RAG:
        - Reduces hallucinations (LLM grounded in real data)
        - Can use up-to-date information (vector DB can be updated anytime)
        - Can cite sources (you know which documents were retrieved)
        - More cost-effective than fine-tuning for knowledge injection
        """,
        metadata={"source": "rag_docs", "topic": "rag"}
    ),
    Document(
        page_content="""
        Vector databases store data as high-dimensional numerical vectors called embeddings.
        These embeddings capture semantic meaning — similar concepts have similar vectors.

        Popular vector databases include:
        - ChromaDB: Open source, runs locally, great for development
        - Pinecone: Managed cloud service, great for production
        - Weaviate: Open source with rich filtering capabilities
        - FAISS: Facebook's library, extremely fast for in-memory search
        - Qdrant: Modern, supports filtering alongside vector search

        ChromaDB specifically:
        - Runs in-memory or persists to disk
        - Has a simple Python API
        - Supports metadata filtering
        - Great for prototyping and small-to-medium datasets
        """,
        metadata={"source": "vectordb_docs", "topic": "vector_databases"}
    ),
    Document(
        page_content="""
        LangGraph Checkpointing allows you to persist conversation state across
        multiple turns of a conversation. Without checkpointing, each graph invocation
        starts fresh with no memory of previous messages.

        How checkpointing works:
        1. You create a checkpointer (MemorySaver for in-memory, SQLiteSaver for disk)
        2. You compile the graph with the checkpointer
        3. Each invocation uses a thread_id to identify the conversation
        4. The checkpointer automatically saves state after each node runs
        5. On the next turn, state is loaded from where it left off

        Thread IDs:
        - Each unique thread_id represents a separate conversation
        - Multiple users can each have their own thread_id
        - Same user different conversations = different thread_ids
        - This is how you support multiple users simultaneously
        """,
        metadata={"source": "memory_docs", "topic": "memory"}
    ),
    Document(
        page_content="""
        Conditional edges in LangGraph allow the graph to branch based on the current
        state. This is what makes LangGraph more than just a linear pipeline.

        How conditional edges work:
        - You define a routing function that takes state and returns a string
        - The string maps to a node name in a routing dictionary
        - LangGraph calls the routing function and follows the matching edge

        Example use cases for conditional edges:
        - Route to different nodes based on question type (math vs. factual vs. chat)
        - Retry retrieval if documents are not relevant (loop back)
        - End early if the task is already complete
        - Branch to error handling if something goes wrong
        - Human-in-the-loop: pause and wait for human approval before continuing

        The power of conditional edges is that they turn your chatbot from a
        fixed pipeline into a dynamic decision-making system.
        """,
        metadata={"source": "edges_docs", "topic": "conditional_edges"}
    ),
    Document(
        page_content="""
        Python is a high-level, interpreted programming language known for its
        simplicity and readability. It was created by Guido van Rossum and first
        released in 1991.

        Key features of Python:
        - Simple, readable syntax (uses indentation instead of braces)
        - Dynamically typed (no need to declare variable types)
        - Large standard library ("batteries included")
        - Huge ecosystem of third-party packages (pip install anything)
        - Excellent for AI/ML, web development, automation, data science

        Python is the dominant language for AI and ML development because of
        libraries like NumPy, PyTorch, TensorFlow, scikit-learn, LangChain, and LangGraph.
        """,
        metadata={"source": "python_docs", "topic": "python"}
    ),
]


def setup_vectorstore(persist_directory: str = "./chroma_db") -> Chroma:
    """
    Creates and persists the vector store.
    Run this once before starting the chatbot.

    Steps:
    1. Split documents into smaller chunks
    2. Load embedding model (HuggingFace runs locally, no API key needed)
    3. Embed all chunks and store in ChromaDB
    4. Persist to disk so we don't rebuild every time
    """

    print("Setting up vector store...")

    # STEP 1: Split documents into chunks
    # RecursiveCharacterTextSplitter tries to split on paragraphs first,
    # then sentences, then words — preserving meaningful boundaries.
    # chunk_size=500: each chunk is at most 500 characters
    # chunk_overlap=100: 100 chars of overlap between chunks so context isn't lost
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(DOCUMENTS)
    print(f"  Created {len(chunks)} chunks from {len(DOCUMENTS)} documents")

    # STEP 2: Load embedding model
    # HuggingFaceEmbeddings downloads and runs the model locally.
    # "all-MiniLM-L6-v2" is small, fast, and surprisingly good for semantic search.
    # First run downloads ~90MB model from HuggingFace Hub.
    print("  Loading embedding model (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # STEP 3: Create ChromaDB from chunks
    # This embeds all chunks and stores them with their vectors.
    # persist_directory saves everything to disk so we can reuse it.
    print("  Embedding chunks and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    print(f"  Vector store ready! Stored at: {persist_directory}")
    return vectorstore


def load_vectorstore(persist_directory: str = "./chroma_db") -> Chroma:
    """
    Loads an existing vector store from disk.
    Use this after setup_vectorstore() has been run once.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


if __name__ == "__main__":
    # Run this file directly to set up the vector store:
    # python vectorstore_setup.py
    vectorstore = setup_vectorstore()
    print("\nTest retrieval:")
    results = vectorstore.similarity_search("How does LangGraph work?", k=2)
    for i, doc in enumerate(results, 1):
        print(f"\n  Result {i}: {doc.page_content[:150]}...")
