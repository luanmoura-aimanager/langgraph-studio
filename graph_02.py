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

def call_claude(state: State) -> State:
    print("Chamando Claude...")
    result = llm.invoke(state["user_input"])
    return {"response": result.content}

builder = StateGraph(State)
builder.add_node("call_claude", call_claude)
builder.set_entry_point("call_claude")
builder.add_edge("call_claude", END)

graph = builder.compile()

output = graph.invoke({"user_input": "What is LangGraph in one sentence?", "response": ""})
print("\nResposta:", output["response"])
