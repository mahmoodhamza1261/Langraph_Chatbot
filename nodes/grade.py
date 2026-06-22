"""
NODES/GRADE.PY — Checks whether retrieved documents actually answer the question.

WHY DO WE NEED GRADING?
Vector search isn't perfect. Sometimes it retrieves documents that are
topically related but don't actually answer the question.

Example:
  Question: "What is the maximum chunk size in LangGraph?"
  Retrieved: General text about LangGraph that never mentions chunk sizes
  Result: If we send this to generate, the LLM will hallucinate an answer

With grading, we catch this problem BEFORE generation:
  - If docs are relevant → go to generate
  - If docs are NOT relevant → rewrite the query and try again

THIS IS A ROUTING NODE:
grade_documents doesn't update state directly.
It's used as the routing function in a conditional edge.
It returns a STRING that tells LangGraph which node to go to next.

This is a key LangGraph pattern:
  Some nodes update state (return a dict)
  Some nodes make routing decisions (return a string)
  Routing functions are used in add_conditional_edges()

THE RETRY LIMIT:
We allow max 2 retries. After that we go to generate anyway
(with a "low confidence" flag) because infinite loops are bad.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from state import GraphState

llm = ChatAnthropic(
    model="claude-opus-4-8",
    temperature=0,
    max_tokens=5    # we only need "yes" or "no"
)

grade_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a relevance grader. Determine if the retrieved document chunks
are relevant to answering the user's question.

Relevant means: the documents contain information that helps answer the question.
Not relevant means: the documents are about different topics.

Reply with ONLY: yes (relevant) or no (not relevant)"""
    ),
    (
        "human",
        """Question: {question}

Retrieved documents:
{documents}

Are these documents relevant to answering the question? Reply yes or no."""
    )
])


def grade_documents(state: GraphState) -> str:
    """
    ROUTING FUNCTION: grade_documents

    IMPORTANT: This function returns a STRING, not a dict.
    It's used as the routing function in add_conditional_edges().
    The string it returns maps to the next node name.

    Possible returns:
    - "generate"       → documents are relevant, proceed to answer generation
    - "rewrite_query"  → documents aren't relevant, rewrite and try again
    - "generate"       → retry limit reached, generate with whatever we have

    Input from state:
        - state["query"]: the current question
        - state["documents"]: the retrieved chunks
        - state["retry_count"]: how many retries have happened
    """
    question = state["query"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    print(f"\n[NODE: grade_documents]")
    print(f"  Grading {len(documents)} documents for relevance...")

    # If no documents retrieved at all, definitely rewrite
    if not documents:
        print("  No documents found → rewriting query")
        if retry_count >= 2:
            return "generate"   # give up, generate with no context
        return "rewrite_query"

    # Format documents for the grading prompt
    formatted_docs = "\n\n---\n\n".join(
        f"Document {i+1}:\n{doc[:300]}"    # only first 300 chars per doc
        for i, doc in enumerate(documents)
    )

    # Ask Claude: are these documents relevant?
    chain = grade_prompt | llm
    result = chain.invoke({
        "question": question,
        "documents": formatted_docs
    })

    verdict = result.content.strip().lower()
    is_relevant = "yes" in verdict

    print(f"  Verdict: {'RELEVANT ✓' if is_relevant else 'NOT RELEVANT ✗'}")

    if is_relevant:
        return "generate"
    else:
        # Check retry limit before deciding to rewrite
        if retry_count >= 2:
            print(f"  Max retries ({retry_count}) reached → generating anyway")
            return "generate"
        else:
            print(f"  Retry {retry_count + 1}/2 → rewriting query")
            return "rewrite_query"
