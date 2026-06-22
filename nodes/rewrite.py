"""
NODES/REWRITE.PY — Rewrites the user's query to improve retrieval success.

WHY REWRITING HELPS:
Users don't write search queries — they ask conversational questions.
  User asks: "So how does that graph memory thing work again?"
  Better for search: "LangGraph checkpointing memory persistence state"

The rewrite node transforms conversational questions into better search queries
by making them more specific, adding relevant technical terms, and removing
conversational filler.

THIS IS A LOOP NODE:
After rewriting, we go BACK to the retrieve node with the improved query.
This creates a cycle in the graph: retrieve → grade → rewrite → retrieve
LangGraph fully supports cycles — that's one of its key advantages over
linear chains.

WHAT HAPPENS TO retry_count?
This node increments retry_count by 1.
The grade node checks retry_count and stops retrying after 2 attempts.
Without this limit, a bad question could loop forever.

GRAPH PATH:
retrieve → grade_documents (not relevant) → rewrite_query → retrieve (again)
                                                                      ↑
                                         [cycle! but limited to 2 loops]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from state import GraphState

llm = ChatAnthropic(
    model="claude-opus-4-8",
    temperature=0.3,    # slight creativity helps generate better search queries
    max_tokens=100
)

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a search query optimizer. Convert conversational questions
into precise, keyword-rich search queries that will find relevant technical documents.

Rules:
- Make the query more specific and technical
- Add relevant synonyms or related terms
- Remove conversational filler words
- Keep it concise (under 15 words)
- Return ONLY the improved query, nothing else"""
    ),
    (
        "human",
        """Original question: {question}

Rewrite this as an optimized search query:"""
    )
])


def rewrite_query(state: GraphState) -> dict:
    """
    NODE: rewrite_query

    Rewrites the current query to be better for vector search,
    then increments retry_count so we know how many attempts we've made.

    Input from state:
        - state["query"]: the original (or previously rewritten) query

    Output to state:
        - query: the new, improved search query
        - retry_count: incremented by 1

    After this node, the graph goes BACK to retrieve with the new query.
    This is the loop in our graph.
    """
    original_query = state["query"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[NODE: rewrite_query]")
    print(f"  Original query: {original_query}")

    # Rewrite the query for better retrieval
    chain = rewrite_prompt | llm
    result = chain.invoke({"question": original_query})

    rewritten_query = result.content.strip()
    print(f"  Rewritten query: {rewritten_query}")

    return {
        "query": rewritten_query,               # update query with improved version
        "retry_count": retry_count + 1          # increment retry counter
    }
