"""
CHATBOT.PY — The main entry point. Runs the chatbot in a terminal loop.

THIS FILE SHOWS:
1. How to set up everything (vector store + graph)
2. How to maintain conversation sessions with thread_id
3. How to invoke the graph with proper state
4. How to handle the streaming vs invoke difference

THREAD IDs — HOW MULTI-USER WORKS:
Each unique thread_id = one conversation session.
- User A: thread_id = "session-alice"
- User B: thread_id = "session-bob"
- They each have separate conversation histories stored by the checkpointer.
- Same user starting over = new thread_id

WHAT IS config = {"configurable": {"thread_id": ...}} ?
When you compile a graph with a checkpointer, every .invoke() needs to
know WHICH conversation to load/save state for.
The config dict is how you pass that information.
Without it, LangGraph doesn't know where to find the saved state.
"""

import os
import uuid
from langchain_core.messages import HumanMessage

# First-run setup check
CHROMA_DIR = "./chroma_db"


def setup_if_needed():
    """
    Checks if the vector store exists. If not, creates it.
    This runs automatically on first launch.
    """
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        print("=" * 60)
        print("FIRST RUN: Setting up vector store...")
        print("(This downloads ~90MB embedding model once)")
        print("=" * 60)
        from vectorstore_setup import setup_vectorstore
        setup_vectorstore(CHROMA_DIR)
        print("=" * 60)
        print("Vector store ready!")
        print("=" * 60)
    else:
        print("[SETUP] Vector store found, skipping setup.")


def run_chatbot():
    """
    Main chatbot loop.

    HOW IT WORKS:
    1. Set up vector store if needed
    2. Build the LangGraph application
    3. Generate a unique session ID (thread_id)
    4. Loop: get user input → invoke graph → print response
    5. State is automatically saved after each turn (checkpointer)
    """

    # ── Setup ────────────────────────────────────────────────────────────
    setup_if_needed()

    print("\n[SETUP] Building graph...")
    from graph import build_graph
    app = build_graph()

    # Generate a unique session ID for this conversation
    # In a real app, this might be the user's ID from your auth system
    session_id = str(uuid.uuid4())[:8]

    # config is passed to EVERY graph.invoke() call
    # It tells the checkpointer which conversation thread to use
    config = {"configurable": {"thread_id": f"session-{session_id}"}}

    # ── Welcome ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  LANGGRAPH RAG CHATBOT")
    print("=" * 60)
    print(f"  Session ID: {session_id}")
    print()
    print("  I can help you with:")
    print("  • Questions about LangGraph, RAG, Python, AI concepts")
    print("  • Math calculations")
    print("  • General knowledge questions")
    print()
    print("  Commands: 'quit' to exit | 'new' for new session")
    print("=" * 60)
    print()

    # ── Main conversation loop ────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Handle special commands
        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye! Thanks for chatting.")
            break

        if user_input.lower() == "new":
            # Start a fresh session
            session_id = str(uuid.uuid4())[:8]
            config = {"configurable": {"thread_id": f"session-{session_id}"}}
            print(f"\n[New session started: {session_id}]\n")
            continue

        # ── Invoke the graph ──────────────────────────────────────────────
        # This is the key call. We pass:
        # 1. The initial state (just the new user message)
        # 2. The config (which session/thread to use)
        #
        # LangGraph will:
        # a) Load saved state from checkpointer (for this thread_id)
        # b) Merge our new message into state["messages"] via add_messages
        # c) Run all the nodes in order (classify → route → ... → generate)
        # d) Save the updated state back to checkpointer
        # e) Return the final state

        print()  # blank line before node debug output
        try:
            result = app.invoke(
                {
                    # Only provide the new user message.
                    # The checkpointer automatically loads all previous messages
                    # and merges this new one via the add_messages reducer.
                    "messages":   [HumanMessage(content=user_input)],
                    "documents":  [],
                    "query":      "",
                    "query_type": "",
                    "retry_count": 0,
                    "generation": "",
                },
                config=config
            )

            # Print the final answer
            answer = result.get("generation", "")
            if answer:
                print(f"\nBot: {answer}")
            else:
                # Fallback: get last AI message from history
                messages = result.get("messages", [])
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        print(f"\nBot: {msg.content}")
                        break

        except Exception as e:
            print(f"\n[ERROR] Something went wrong: {e}")
            print("Please try again.")

        print()  # blank line after response


if __name__ == "__main__":
    run_chatbot()
