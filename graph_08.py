# graph_08.py — Conditional Edges (Branching)
#
# LEARNING GOALS:
#   - Use conditional edges to route execution to different nodes at runtime
#   - Understand how a "router" function drives branching logic
#   - See how multiple response nodes can share the same graph with different behavior
#
# NEW CONCEPT — CONDITIONAL EDGES:
#   Until now, every edge was fixed: node_a always went to node_b.
#   Conditional edges let the graph DECIDE which node to run next,
#   based on the current state. This is how you build branching pipelines.
#
#   add_conditional_edges(source, router_fn, mapping) works like this:
#     1. source node runs and updates state
#     2. router_fn(state) is called — it returns a string (the "route key")
#     3. mapping[route_key] gives the name of the next node to run
#
# PIPELINE (branching):
#
#              ┌──── "TECHNICAL" ──→ respond_technical ──→ END
#   classify ──┤
#              └──── "GENERAL"    ──→ respond_general   ──→ END

import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from operator import add

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# --- HELPER ---
# Extracted into its own function so both respond nodes can share the same logic
# without duplicating code. Asks the user whether to print the history, then does so.
def print_historico(messages):
    ver = input("Ver histórico? (s/n): ")
    if ver.lower() == "s":
        print("--- Histórico ---")
        for msg in messages:
            print(f"{type(msg).__name__}: {msg.content}")
        print("----------------")

# --- STATE ---
# Same pattern as graph_07: category is reset each turn, messages accumulate.
class State(TypedDict):
    category: str                   # "TECHNICAL" or "GENERAL" — set by classify
    messages: Annotated[list, add]  # full conversation history, grows over time

# --- NODE 1: classify ---
# Reads only the last message (efficient — no need to classify the whole history),
# then sets the category that the router will use to decide the next node.
def classify(state: State) -> State:
    last_message = state["messages"][-1].content
    messages = [
        SystemMessage(content="Classify the user question into exactly one word: TECHNICAL or GENERAL."),
        HumanMessage(content=last_message)
    ]
    result = llm.invoke(messages)
    category = result.content.strip().upper()
    # Normalize: if the model said anything with "TECHNICAL", use that; else GENERAL
    category = "TECHNICAL" if "TECHNICAL" in category else "GENERAL"
    print(f"Categoria: {category}")
    return {"category": category}

# --- ROUTER FUNCTION ---
# This is NOT a node — it's a plain function called by LangGraph to decide routing.
# It receives the current state and must return a string that matches a key
# in the mapping passed to add_conditional_edges().
# Keep router functions simple: just read state and return a string.
def router(state: State) -> str:
    return state["category"]  # returns "TECHNICAL" or "GENERAL"

# --- NODE 2a: respond_technical ---
# Runs when the router returns "TECHNICAL".
# Uses a technical expert system prompt for a more detailed, precise answer.
def respond_technical(state: State) -> State:
    print("→ Caminho TECHNICAL")
    messages = [
        SystemMessage(content="You are a technical expert. Give a detailed, precise answer in Portuguese."),
    ] + state["messages"]  # include full history for context
    result = llm.invoke(messages)
    print(f"Claude: {result.content}")

    # Delegate history display to the shared helper
    print_historico(state["messages"])

    return {"messages": [result]}

# --- NODE 2b: respond_general ---
# Runs when the router returns "GENERAL".
# Uses a friendly, casual system prompt — same logic, different persona.
def respond_general(state: State) -> State:
    print("→ Caminho GENERAL")
    messages = [
        SystemMessage(content="You are a friendly assistant. Give a simple, casual answer in Portuguese."),
    ] + state["messages"]
    result = llm.invoke(messages)
    print(f"Claude: {result.content}")

    # Delegate history display to the shared helper
    print_historico(state["messages"])

    return {"messages": [result]}

# --- GRAPH ASSEMBLY ---
builder = StateGraph(State)

# Register all three nodes
builder.add_node("classify", classify)
builder.add_node("respond_technical", respond_technical)
builder.add_node("respond_general", respond_general)

builder.set_entry_point("classify")

# add_conditional_edges replaces a plain add_edge after "classify".
# Instead of always going to the same node, LangGraph calls router(state)
# and uses the returned string to look up the next node in the dict.
builder.add_conditional_edges(
    "classify",          # source node
    router,              # function that returns the route key
    {
        "TECHNICAL": "respond_technical",  # if router returns "TECHNICAL" → this node
        "GENERAL": "respond_general",      # if router returns "GENERAL"   → this node
    }
)

# Both branches eventually reach END
builder.add_edge("respond_technical", END)
builder.add_edge("respond_general", END)

# Compile with MemorySaver so history persists across turns per user
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- CHAT LOOP ---
print("Chat com branching! Digite 'sair' para encerrar.\n")
while True:
    thread = input("Nome do usuário: ")
    # thread_id isolates each user's conversation history
    config = {"configurable": {"thread_id": thread.lower()}}

    user_input = input("Digite sua mensagem: ")
    if user_input.lower() == "sair":
        break

    graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
