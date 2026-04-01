# graph_02.py — Single Node with Claude
#
# LEARNING GOALS:
#   - Integrate a real LLM (Claude Haiku) into a LangGraph node
#   - Understand how ChatAnthropic wraps the API call
#   - See the minimal setup needed to use an LLM inside a graph
#
# NEW CONCEPT: Instead of doing string manipulation in our nodes,
# we now call an LLM and store its response in the state.

import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

# --- LLM SETUP ---
# ChatAnthropic is LangChain's wrapper around the Anthropic API.
# We specify the model and pass the API key from an environment variable
# (never hardcode secrets in source code!).
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# --- STATE ---
# Two fields: the user's question goes in, the LLM's answer comes out.
class State(TypedDict):
    user_input: str   # the question we want to ask Claude
    response: str     # will be filled with Claude's answer

# --- NODE ---
# This node takes the user's input string and passes it directly to Claude.
# llm.invoke() sends the request and returns an AIMessage object.
# We extract the text content with .content and store it in state.
def call_claude(state: State) -> State:
    print("Chamando Claude...")
    result = llm.invoke(state["user_input"])  # returns an AIMessage
    return {"response": result.content}       # .content is the text string

# --- GRAPH ASSEMBLY ---
# The simplest possible graph: one node, goes straight to END.
builder = StateGraph(State)
builder.add_node("call_claude", call_claude)
builder.set_entry_point("call_claude")
builder.add_edge("call_claude", END)

graph = builder.compile()

# --- RUN ---
# We pass in a fixed question to keep this example self-contained.
# Notice "response" starts empty — the node fills it after calling Claude.
output = graph.invoke({"user_input": "What is LangGraph in one sentence?", "response": ""})
print("\nResposta:", output["response"])
