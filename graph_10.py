# graph_10.py — Multiple Isolated Personas via Thread IDs
#
# What this file teaches:
#   - How a single compiled graph can host completely separate conversations at the same time.
#   - Each thread_id is an isolated "session": different personas, different histories.
#   - The same node code runs for every thread, but the loaded state is always thread-specific.
#   - How to inspect the saved state of individual threads after the graph has run.
#
# Key insight: thread_id is the only thing that separates two users (or two personas).
# Changing the thread_id is all it takes to start a fresh conversation on the same graph.

from typing import Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

# add_messages tells LangGraph to APPEND new messages, not replace the list
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

def chat(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chat", chat)
builder.set_entry_point("chat")
builder.add_edge("chat", END)

# One MemorySaver stores all threads. Each thread_id gets its own isolated history inside it.
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# --- Thread 1: Bob the Kitchen Chef ---
# Turn 1 establishes a persona for this thread. The model is told it's "Bob".
# Turn 2 takes advantage of that context — the model stays in character.
print("=== Thread 1: Kitchen Chef ===")
config = {"configurable": {"thread_id": "thread-1"}}
graph.invoke(
    {"messages": [{"role": "user", "content": "You're Bob. My personal kitchen chef."}]},
    config=config)
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Introduce yourself and give me a recipe of a 'bruaca' (famous brazilian food similar to pancake)."}]},
    config=config
)
print(result["messages"][-1].content)

# --- Thread 2: Alex the Football Expert ---
# Completely separate history. "Alex" has no knowledge of "Bob" or thread-1.
# Changing thread_id is the only difference — the graph, nodes, and LLM are identical.
print("=== Thread 2: Football Expert ===")
config = {"configurable": {"thread_id": "thread-2"}}
graph.invoke(
    {"messages": [{"role": "user", "content": "You're Alex. My football expert."}]},
    config=config)
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Introduce yourself and tell me the difference between soccer and football."}]},
    config=config
)
print(result["messages"][-1].content)

# --- Inspect saved state per thread ---
# get_state() retrieves the full message history for a given thread.
# The [:60] truncation here is just for readable terminal output.
# In a real app you'd use the full message content.
print("\n=== Full state thread-1 ===")
for msg in graph.get_state({"configurable": {"thread_id": "thread-1"}}).values["messages"]:
    print(f"{msg.type}: {msg.content[:60]}")

print("\n=== Full state thread-2 ===")
for msg in graph.get_state({"configurable": {"thread_id": "thread-2"}}).values["messages"]:
    print(f"{msg.type}: {msg.content[:60]}")
