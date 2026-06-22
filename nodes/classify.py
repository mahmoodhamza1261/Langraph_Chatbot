"""
NODES/CLASSIFY.PY — The first node that runs for every user message.

WHY DO WE NEED CLASSIFICATION?
Different questions need different handling:
  - "Hello!" → just respond warmly, no need to search anything
  - "What is 234 * 17?" → use a calculator for accuracy
  - "What does the LangGraph doc say about state?" → search the vector store (RAG)
  - "Explain how TCP/IP works" → just use LLM's knowledge directly

Without classification, you'd either run RAG on EVERY message (slow, wasteful)
or skip RAG entirely (can't answer questions about your docs).

HOW THIS NODE WORKS:
1. Read the latest user message from state
2. Send it to Claude with a strict classification prompt
3. Get back one word: greeting | math | rag | general
4. Store that word in state["query_type"]
5. The conditional edge AFTER this node reads query_type to decide routing

WHAT IS A NODE?
A node is just a plain Python function:
  - Takes GraphState as input
  - Returns a dict with only the keys it wants to update
  - LangGraph merges this dict into the full state automatically
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from state import GraphState

# We create the LLM once at module level (not inside the function)
# so it's not re-initialized on every call — much more efficient
llm = ChatAnthropic(
    model="claude-opus-4-8",
    temperature=0,        # temperature=0 means deterministic output
    max_tokens=10         # we only need ONE word, so limit tokens
)

# The prompt that tells Claude how to classify the query
# Note: we use ChatPromptTemplate for proper message formatting
classify_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query classifier. Your ONLY job is to classify user questions.
Reply with EXACTLY ONE WORD from these options:

- greeting  → Hello, hi, thanks, bye, how are you, compliments, small talk
- math      → Any calculation: arithmetic, algebra, percentages, unit conversion
- rag       → Questions about LangGraph, RAG, Python, vector databases, AI concepts
- general   → Everything else: history, science, coding help, explanations

Do not explain. Do not add punctuation. Reply with ONE WORD ONLY."""
    ),
    (
        "human",
        "Classify this question: {question}"
    )
])


def classify_query(state: GraphState) -> dict:
    """
    NODE: classify_query

    This is the ENTRY POINT of our graph. Every conversation turn starts here.

    Input from state:
        - state["messages"]: the full conversation history
          We grab the LAST message ([-1]) which is the current user question

    Output to state:
        - query: the raw text of the user's question
        - query_type: "greeting" | "math" | "rag" | "general"
        - retry_count: reset to 0 for each new question
        - documents: reset to empty list for each new question
    """
    # Get the current user question (last message in history)
    question = state["messages"][-1].content

    print(f"\n{'='*50}")
    print(f"[NODE: classify_query]")
    print(f"  Question: {question}")

    # Call Claude to classify the question
    chain = classify_prompt | llm
    result = chain.invoke({"question": question})

    # Clean up the response (strip whitespace, lowercase)
    query_type = result.content.strip().lower()

    # Safety: if Claude returns something unexpected, default to "general"
    valid_types = {"greeting", "math", "rag", "general"}
    if query_type not in valid_types:
        print(f"  WARNING: Unexpected classification '{query_type}', defaulting to 'general'")
        query_type = "general"

    print(f"  Classified as: {query_type}")

    # Return only the keys we're updating
    # LangGraph will merge this with the existing state
    return {
        "query": question,
        "query_type": query_type,
        "retry_count": 0,       # reset retry counter for new question
        "documents": [],         # reset documents for new question
        "generation": ""         # reset previous answer
    }
