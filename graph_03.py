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
    print("=== DENTRO DO NÓ ===")
    print("State recebido:", state)
    print("Tipo do state:", type(state))
    print("user_input:", state["user_input"])
    print()

    result = llm.invoke(state["user_input"])

    print("=== OBJETO RETORNADO PELO LLM ===")
    print("result completo:", result)
    print("Tipo do result:", type(result))
    print("result.content:", result.content)
    print("result.response_metadata:", result.response_metadata)
    print("result.id:", result.id)
    print()

    return {"response": result.content}

builder = StateGraph(State)
builder.add_node("call_claude", call_claude)
builder.set_entry_point("call_claude")
builder.add_edge("call_claude", END)

graph = builder.compile()

print("=== GRAPH COMPILADO ===")
print("Tipo do graph:", type(graph))
print()

print("=== CHAMANDO graph.invoke() ===")
output = graph.invoke({"user_input": "What is LangGraph in one sentence?", "response": ""})

print("=== OUTPUT FINAL ===")
print("output completo:", output)
print("Tipo do output:", type(output))
print("output['response']:", output["response"])
