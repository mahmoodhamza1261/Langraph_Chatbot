"""
NODES/GENERATE.PY — Generates the final answer using the LLM + retrieved context.

THIS IS THE MOST IMPORTANT NODE for RAG and general questions.

TWO MODES:
1. RAG mode (documents available):
   - Formats documents as context
   - Tells Claude to answer from the context
   - Reduces hallucinations

2. General mode (no documents):
   - Just sends the conversation history to Claude
   - Claude uses its training knowledge
   - For general knowledge questions

WHY INCLUDE CHAT HISTORY?
Multi-turn conversation. If the user asks:
  Turn 1: "What is LangGraph?"
  Turn 2: "Can you give an example?"

In turn 2, "it" refers to LangGraph. Without chat history, Claude would
ask "example of what?" — frustrating! With full messages history, Claude
understands the context.

SYSTEM PROMPT DESIGN:
The system prompt is crucial. It tells Claude:
- WHO it is (a helpful assistant)
- WHAT DATA to use (only the context, not made-up info)
- HOW to respond (cite sources, be concise)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, AIMessage
from state import GraphState

llm = ChatAnthropic(
    model="claude-opus-4-8",
    temperature=0.7,    # some creativity in responses is fine here
    max_tokens=1024
)

# RAG system prompt: uses retrieved documents as context
RAG_SYSTEM_PROMPT = """You are a knowledgeable AI assistant with access to a specialized knowledge base.

INSTRUCTIONS:
1. Answer the user's question using the provided context documents
2. If the context contains the answer, use it and be specific
3. If the context only partially answers the question, say what you found and what you don't know
4. If the context doesn't contain the answer at all, say so clearly — don't hallucinate
5. Keep answers focused and well-structured

CONTEXT DOCUMENTS:
{context}
"""

# General system prompt: no documents, just LLM knowledge
GENERAL_SYSTEM_PROMPT = """You are a helpful, knowledgeable AI assistant.
Answer the user's questions accurately and concisely.
If you're not sure about something, say so."""


def generate(state: GraphState) -> dict:
    """
    NODE: generate

    The final answer generation node. Used for both:
    - RAG questions (with retrieved documents in state)
    - General questions (no documents, just LLM knowledge)

    Input from state:
        - state["messages"]: full conversation history (for multi-turn context)
        - state["documents"]: retrieved chunks (may be empty for general questions)

    Output to state:
        - generation: the final answer text
        - messages: the AI response appended to conversation history

    HOW IT USES FULL MESSAGE HISTORY:
    We pass state["messages"] (all prior turns) along with the system prompt.
    This gives Claude the full conversation context so follow-up questions work.
    """
    messages = state["messages"]
    documents = state.get("documents", [])

    print(f"\n[NODE: generate]")
    print(f"  Documents in context: {len(documents)}")
    print(f"  Message history length: {len(messages)}")

    # Choose system prompt based on whether we have retrieved documents
    if documents:
        # RAG mode: format documents and inject into system prompt
        context = "\n\n---\n\n".join(
            f"[Document {i+1}]:\n{doc}"
            for i, doc in enumerate(documents)
        )
        system_content = RAG_SYSTEM_PROMPT.format(context=context)
        print(f"  Mode: RAG (using {len(documents)} documents)")
    else:
        # General mode: no documents, just Claude's knowledge
        system_content = GENERAL_SYSTEM_PROMPT
        print(f"  Mode: General knowledge")

    # Build the full message list for Claude:
    # [SystemMessage with instructions] + [all prior HumanMessage/AIMessage turns]
    # This is the key to multi-turn conversation — include ALL history
    full_messages = [SystemMessage(content=system_content)] + list(messages)

    # Call Claude with full context
    response = llm.invoke(full_messages)

    answer = response.content.strip()
    print(f"  Generated answer: {answer[:100]}...")

    return {
        "generation": answer,
        # add_messages reducer will APPEND this to existing messages
        # so next turn still has full history
        "messages": [AIMessage(content=answer)]
    }
