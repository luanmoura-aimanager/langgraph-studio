# graph_09.py — Persistent Memory + Custom State Fields
#
# What this file teaches:
#   - How to add non-message fields to state (e.g. a turn counter) alongside the message list.
#   - The difference between fields that ACCUMULATE (messages) and fields that OVERWRITE (turn_count).
#   - How the same graph and checkpointer can host multiple independent threads.
#   - How to inspect the full conversation state after the graph has run.
#
# Key insight: State can hold any fields you need. `messages` uses `add_messages` to append,
# but `turn_count` is a plain int — each return value simply replaces the previous one.
# LangGraph merges returned fields into state individually, so the two strategies coexist.

from typing import Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

# add_messages tells LangGraph to APPEND new messages, not replace the list.
# turn_count is a plain int — it gets OVERWRITTEN on each turn (no reducer needed).
class State(TypedDict):
    messages: Annotated[list, add_messages]
    turn_count: int

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

def chat(state: State):
    response = llm.invoke(state["messages"])
    # state.get() with a default handles the very first turn when turn_count is not yet set.
    current = state.get("turn_count", 0)
    # Returning both fields: messages is APPENDED (via add_messages reducer),
    # turn_count is REPLACED with the new value.
    return {"messages": [response], "turn_count": current + 1}

builder = StateGraph(State)
builder.add_node("chat", chat)
builder.set_entry_point("chat")
builder.add_edge("chat", END)

# MemorySaver stores a snapshot of state after every invocation.
# On the next invoke() with the same thread_id, LangGraph restores that snapshot
# before running the graph — so the node sees the full message history automatically.
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# --- same thread, two messages ---
# The thread_id ties invocations together. Both calls below share the same memory.
config = {"configurable": {"thread_id": "thread-1"}}

print("=== Turn 1 ===")
result = graph.invoke(
    {"messages": [{"role": "user", "content": "My name is Misael."}]},
    config=config
)
print(result["messages"][-1].content)
print(result["turn_count"])  # Should print 1

print("\n=== Turn 2 ===")
# We only pass the NEW user message — LangGraph loads and merges the prior turn automatically.
result = graph.invoke(
    {"messages": [{"role": "user", "content": "What is my name?"}]},
    config=config
)
print(result["messages"][-1].content)   # Should recall "Misael"
print(result["turn_count"])             # Should print 2

# get_state() lets you inspect the full saved state for any thread at any time.
# This is useful for debugging: you can see every message that was exchanged.
print("\n=== Full state after Turn 2 ===")
state = graph.get_state(config)
for msg in state.values["messages"]:
    print(f"{msg.type}: {msg.content}")

# Commented-out block below (thread-2) would show that a NEW thread starts with a blank slate —
# the model would not know the user's name. Uncomment to verify thread isolation.
# print("\n=== New thread ===")
# new_config = {"configurable": {"thread_id": "thread-2"}}
# result = graph.invoke(
#     {"messages": [{"role": "user", "content": "What is my name?"}]},
#     config=new_config
# )
# print(result["messages"][-1].content)

# --- Different thread: accumulates a different user profile ---
# thread-3 is completely independent of thread-1. It has its own message history
# and its own turn_count, all stored separately in MemorySaver.
print("\n=== New thread ===")
new_config = {"configurable": {"thread_id": "thread-3"}}

graph.invoke({"messages": [{"role": "user", "content": "I'm from Fortaleza."}]}, config=new_config)
graph.invoke({"messages": [{"role": "user", "content": "I'm a Data Scientist."}]}, config=new_config)

# After three turns in thread-3, the model has enough context to summarize the user.
result = graph.invoke({"messages": [{"role": "user", "content": "Summarize what you know about me."}]}, config=new_config)
print(result["messages"][-1].content)
print(result["turn_count"])  # Should print 3 (three turns in this thread)
