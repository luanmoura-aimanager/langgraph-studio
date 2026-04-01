# graph_01.py — Basic StateGraph (no LLM)
#
# LEARNING GOALS:
#   - Understand what a "State" is and why every graph needs one
#   - See how nodes are just regular Python functions
#   - Understand how edges define the execution order between nodes
#
# KEY CONCEPT: A LangGraph graph is a pipeline where data (called "state")
# flows through a sequence of functions (called "nodes"), connected by edges.

from typing import TypedDict
from langgraph.graph import StateGraph, END

# --- STATE ---
# The State is a shared "notebook" that all nodes can read from and write to.
# We define it as a TypedDict so Python knows which keys exist and their types.
#
# Every node receives the current state and returns a dict with
# only the keys it wants to update — the rest stay unchanged.
class State(TypedDict):
    message: str  # the initial input we pass into the graph
    result: str   # will be built up progressively as nodes run

# --- NODES ---
# A node is a plain Python function with this signature:
#   input:  the current state (a dict)
#   output: a dict with the keys you want to update
#
# You don't need to return the full state — just the fields you changed.
def node_a(state: State) -> State:
    print("Node A executando...")
    # Reads "message" from state, writes a new value into "result"
    return {"result": f"Node A processou: '{state['message']}'"}

def node_b(state: State) -> State:
    print("Node B executando...")
    # Reads "result" (already set by node_a) and appends its own stamp.
    # This illustrates how nodes can build on the outputs of previous nodes.
    return {"result": state["result"] + " → Node B também passou aqui"}

# --- GRAPH ASSEMBLY ---
# StateGraph(State) creates a builder — a blueprint for our pipeline.
# We first add nodes, then connect them with edges to define execution order.
builder = StateGraph(State)

builder.add_node("node_a", node_a)    # register function as a named node
builder.add_node("node_b", node_b)

builder.set_entry_point("node_a")     # execution always starts at this node
builder.add_edge("node_a", "node_b")  # after node_a finishes → run node_b
builder.add_edge("node_b", END)       # END is a special marker: "stop here"

# compile() validates the graph structure and returns a runnable object
graph = builder.compile()

# --- RUN ---
# invoke() starts the graph with an initial state and returns the final state.
# "result" starts empty — the nodes fill it in as they execute.
output = graph.invoke({"message": "olá LangGraph", "result": ""})
print("\nEstado final:", output)
