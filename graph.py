"""
GRAPH.PY — The heart of the application. Connects all nodes with edges.

KEY CONCEPTS:
1. StateGraph(GraphState) — graph typed to your state
2. add_node(name, function) — registers a node
3. add_edge(A, B) — fixed A → B connection
4. add_conditional_edges(A, routing_fn, mapping) — branch based on function return value
5. compile(checkpointer=...) — build the runnable + enable memory

TWO TYPES OF FUNCTIONS IN LANGGRAPH:
- Node functions    → take state, return dict (updates state)
- Routing functions → take state, return string (decides next node)

grade_documents is a routing function — it returns "generate" or "rewrite_query".
LangGraph passes routing functions directly to add_conditional_edges.
No wrapper or extra node needed.

COMPLETE GRAPH FLOW:
START
  └─► classify_query
            │
     ┌──────┼──────────┬──────────┐
 "greeting" "math"    "rag"   "general"
     │        │         │          │
     ▼        ▼         ▼          ▼
 respond   calculator  retrieve  generate
     │        │         │          │
    END      END   grade_documents  │
                    │         │    │
               "generate" "rewrite"│
                    │         │    │
                 generate  rewrite_query
                    │         │
                   END    retrieve  ← LOOP (max 2 retries)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import GraphState
from nodes.classify import classify_query
from nodes.retrieve import retrieve
from nodes.grade import grade_documents       # routing function: returns "generate" or "rewrite_query"
from nodes.rewrite import rewrite_query
from nodes.generate import generate
from nodes.tools import use_calculator, respond_directly


def route_after_classify(state: GraphState) -> str:
    """
    Routing function called after classify_query.
    Reads state["query_type"] and returns it — LangGraph uses this
    to look up which node to go to next in the mapping dict below.
    """
    query_type = state["query_type"]
    print(f"\n[ROUTER] Classify result → '{query_type}'")
    return query_type


def build_graph():
    """
    Assembles and compiles the full chatbot graph.
    Returns a runnable app with memory checkpointing.
    """

    # ── Create the graph with our state type ──────────────────────────────
    workflow = StateGraph(GraphState)

    # ── Register every node ───────────────────────────────────────────────
    workflow.add_node("classify_query",  classify_query)
    workflow.add_node("retrieve",        retrieve)
    workflow.add_node("rewrite_query",   rewrite_query)
    workflow.add_node("generate",        generate)
    workflow.add_node("use_calculator",  use_calculator)
    workflow.add_node("respond_directly", respond_directly)

    # ── Entry point ───────────────────────────────────────────────────────
    workflow.add_edge(START, "classify_query")

    # ── After classify: branch into 4 paths ───────────────────────────────
    # route_after_classify returns state["query_type"] as a string.
    # LangGraph maps that string to the next node via the dict below.
    workflow.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "greeting": "respond_directly",
            "math":     "use_calculator",
            "rag":      "retrieve",
            "general":  "generate",
        }
    )

    # ── After retrieve: grade documents (routing function used directly) ───
    # grade_documents takes state, returns "generate" or "rewrite_query".
    # Because it returns a string, it IS the routing function.
    # No wrapper or extra node needed — this is the correct LangGraph pattern.
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,            # called by LangGraph, reads state, returns string
        {
            "generate":      "generate",
            "rewrite_query": "rewrite_query",
        }
    )

    # ── Loop: rewrite → retrieve (creates the retry cycle) ────────────────
    # retry_count in state prevents infinite loops (max 2 retries in grade.py)
    workflow.add_edge("rewrite_query", "retrieve")

    # ── Terminal edges ─────────────────────────────────────────────────────
    workflow.add_edge("generate",        END)
    workflow.add_edge("use_calculator",  END)
    workflow.add_edge("respond_directly", END)

    # ── Compile with memory checkpointing ─────────────────────────────────
    # MemorySaver stores conversation state keyed by thread_id.
    # Each unique thread_id = one separate conversation session.
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    print("[GRAPH] Compiled successfully.")
    return app


if __name__ == "__main__":
    app = build_graph()
    print("Graph OK — run chatbot.py to start chatting.")
