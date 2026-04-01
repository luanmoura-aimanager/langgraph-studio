# graph_06.py — Persistent Memory Chat
#
# LEARNING GOALS:
#   - Use MemorySaver to persist conversation history across turns
#   - Understand Annotated[list, add] for state fields that accumulate
#   - Learn how thread_id creates separate memory contexts per user
#
# NEW CONCEPTS:
#   - MemorySaver: a built-in LangGraph checkpointer that stores state in memory
#   - Annotated[list, add]: tells LangGraph to *append* new messages instead of
#     overwriting them — this is how conversation history is maintained
#   - thread_id: a string key that isolates one user's history from another's
#
# HOW MEMORY WORKS:
#   On the first invoke(), the graph starts with an empty message list.
#   On every subsequent invoke() with the same thread_id, LangGraph loads
#   the previous state from the checkpointer and merges the new messages in.
#   The node then sees the *full conversation history*, not just the latest message.

import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from operator import add

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# --- STATE ---
# Annotated[list, add] is the key pattern for persistent message lists.
# Without it, each invoke() would overwrite the messages field.
# With it, LangGraph uses Python's `add` (list concatenation) to merge
# the new messages returned by a node with the existing history.
class State(TypedDict):
    messages: Annotated[list, add]  # grows across turns; never overwritten

# --- NODE ---
def chat(state: State) -> State:
    # state["messages"] contains the FULL history for this thread_id,
    # including all previous turns — MemorySaver loaded it automatically.
    print(f"\nHistórico tem {len(state['messages'])} mensagens")

    # Prepend a system prompt to the full history before sending to Claude.
    # The system message is NOT stored in state — it's added fresh each time.
    all_messages = [
        SystemMessage(content="You are a helpful assistant. Answer in Portuguese.")
    ] + state["messages"]

    print("--- Histórico ---")
    ver_historico = input("Ver histórico? (s/n): ")
    if ver_historico.lower() == "s":
        for msg in state["messages"]:
            # type(msg).__name__ prints "HumanMessage" or "AIMessage"
            print(f"{type(msg).__name__}: {msg.content}")
    print("----------------")

    result = llm.invoke(all_messages)
    print(f"Claude: {result.content}")

    # Return only the new AI message — Annotated[list, add] will append it
    # to the existing history automatically.
    return {"messages": [result]}

# --- GRAPH ASSEMBLY ---
builder = StateGraph(State)
builder.add_node("chat", chat)
builder.set_entry_point("chat")
builder.add_edge("chat", END)

# MemorySaver is the checkpointer — it saves and loads state between invocations.
# Pass it to compile() so every invoke() with a matching thread_id gets its history.
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- CHAT LOOP ---
print("Chat com memória! Digite 'sair' para encerrar.\n")
while True:
    thread = input("Nome do usuário: ")
    # config carries the thread_id — this is how LangGraph knows which
    # saved state to load. Different names = different conversation histories.
    config = {"configurable": {"thread_id": thread.lower()}}

    user_input = input("Digite sua mensagem: ")
    if user_input.lower() == "sair":
        break

    # Wrap the user's text in a HumanMessage so it fits the messages list format.
    graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
