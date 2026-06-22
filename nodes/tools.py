"""
NODES/TOOLS.PY — Specialized nodes for math and greetings.

WHY SEPARATE NODES FOR THESE?
Math: LLMs are notoriously bad at arithmetic. 17 * 23 = ? LLMs often get
      this wrong. By extracting the math expression and using Python's eval(),
      we get 100% accurate answers. Then we let the LLM format a nice response.

Greetings: No need to search documents or call Claude multiple times.
           Just respond directly with a warm message. Fast and efficient.

SECURITY NOTE ON eval():
We NEVER call eval() directly on user input — that's a security disaster.
Instead we:
1. Ask Claude to extract ONLY the math expression (numbers and operators)
2. Validate it contains only safe characters
3. Then call eval() on the sanitized expression

This is a safe eval pattern for math-only use cases.
"""

import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from state import GraphState

llm = ChatAnthropic(
    model="claude-opus-4-8",
    temperature=0,
    max_tokens=50
)

# ─── MATH HANDLING ──────────────────────────────────────────────────────────

extract_math_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Extract ONLY the mathematical expression from the question.
Return ONLY the expression with standard math operators: + - * / ** () %
Do NOT include words, units, or explanations.
Examples:
  "What is 5 plus 3?" → 5 + 3
  "Calculate 15% of 200" → 200 * 0.15
  "What's 2 to the power of 8?" → 2 ** 8"""
    ),
    ("human", "{question}")
])


def use_calculator(state: GraphState) -> dict:
    """
    NODE: use_calculator

    Handles math questions with 100% accuracy by using Python's eval().

    Flow:
    1. Ask Claude to extract the math expression from natural language
    2. Validate the expression is safe (only math characters)
    3. Evaluate with Python's eval()
    4. Format a nice response and add it to state

    Input from state:
        - state["query"]: the math question in natural language

    Output to state:
        - generation: the calculated answer as a formatted string
        - messages: the AI response appended to chat history
    """
    question = state["query"]

    print(f"\n[NODE: use_calculator]")
    print(f"  Question: {question}")

    try:
        # Step 1: Extract the math expression using Claude
        chain = extract_math_prompt | llm
        result = chain.invoke({"question": question})
        expression = result.content.strip()

        print(f"  Extracted expression: {expression}")

        # Step 2: Validate — only allow safe math characters
        # This prevents code injection attacks
        safe_pattern = r'^[\d\s\+\-\*\/\(\)\.\%\*\^]+$'
        if not re.match(safe_pattern, expression):
            raise ValueError(f"Unsafe expression: {expression}")

        # Step 3: Calculate using Python
        # We use eval() ONLY on the validated, extracted expression
        calculated_result = eval(expression)  # noqa: S307

        # Step 4: Format a nice response
        answer = f"The answer is: **{expression} = {calculated_result}**"
        print(f"  Result: {calculated_result}")

    except Exception as e:
        print(f"  Calculator error: {e}")
        answer = (
            f"I couldn't calculate that precisely. "
            f"Here's my best estimate for '{question}': "
            f"Please rephrase as a clear math expression."
        )

    return {
        "generation": answer,
        "messages": [AIMessage(content=answer)]
    }


# ─── GREETING HANDLING ──────────────────────────────────────────────────────

greeting_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly, helpful AI assistant.
Respond warmly to greetings and small talk.
Keep responses short and friendly (2-3 sentences max).
Mention that you can:
- Answer questions from your knowledge base (RAG)
- Help with math calculations
- Answer general knowledge questions"""
    ),
    ("human", "{question}")
])


def respond_directly(state: GraphState) -> dict:
    """
    NODE: respond_directly

    Handles greetings and small talk without any document retrieval.
    Fast path: no vector search, just a direct LLM response.

    Input from state:
        - state["query"]: the greeting or small talk message

    Output to state:
        - generation: the friendly response
        - messages: the AI response appended to chat history
    """
    question = state["query"]

    print(f"\n[NODE: respond_directly]")
    print(f"  Greeting: {question}")

    chain = greeting_prompt | llm
    result = chain.invoke({"question": question})

    answer = result.content.strip()
    print(f"  Response: {answer[:80]}...")

    return {
        "generation": answer,
        "messages": [AIMessage(content=answer)]
    }
