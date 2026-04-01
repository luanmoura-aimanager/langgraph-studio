# graph_03.py — Inspecting the LLM Response Object
#
# LEARNING GOALS:
#   - Explore the full AIMessage object returned by an LLM call
#   - Understand what data is available beyond just .content
#   - Learn to inspect state and graph objects at runtime
#
# NEW CONCEPT: The LLM doesn't return a plain string — it returns a rich
# AIMessage object with metadata you can use for debugging, billing tracking,
# caching, and more. This file prints everything so you can see it all.

import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

class State(TypedDict):
    user_input: str
    response: str

# --- NODE (with debug output) ---
def call_claude(state: State) -> State:
    # --- Inspect the incoming state ---
    # This shows you exactly what the node receives when it runs.
    print("=== DENTRO DO NÓ ===")
    print("State recebido:", state)
    print("Tipo do state:", type(state))
    print("user_input:", state["user_input"])
    print()

    result = llm.invoke(state["user_input"])

    # --- Inspect the AIMessage object ---
    # .content         → the actual text response (what we usually care about)
    # .response_metadata → token counts, stop reason, model name, etc.
    # .id              → a unique identifier for this specific LLM call
    print("=== OBJETO RETORNADO PELO LLM ===")
    print("result completo:", result)
    print("Tipo do result:", type(result))
    print("result.content:", result.content)
    print("result.response_metadata:", result.response_metadata)
    print("result.id:", result.id)
    print()

    return {"response": result.content}

# --- GRAPH ASSEMBLY ---
# Same single-node structure as graph_02 — the new learning is all in the node.
builder = StateGraph(State)
builder.add_node("call_claude", call_claude)
builder.set_entry_point("call_claude")
builder.add_edge("call_claude", END)

graph = builder.compile()

# --- Inspect the compiled graph object ---
# Useful to know: the compiled graph is a CompiledStateGraph, not a plain dict.
print("=== GRAPH COMPILADO ===")
print("Tipo do graph:", type(graph))
print()

# --- RUN ---
print("=== CHAMANDO graph.invoke() ===")
output = graph.invoke({"user_input": "What is LangGraph in one sentence?", "response": ""})

# --- Inspect the final output ---
# output is just a regular Python dict — the final state after all nodes ran.
print("=== OUTPUT FINAL ===")
print("output completo:", output)
print("Tipo do output:", type(output))
print("output['response']:", output["response"])
