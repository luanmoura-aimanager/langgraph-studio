import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

class State(TypedDict):
    user_input: str
    category: str
    response: str
    summary: str

def classify(state: State) -> State:
    print("=== NÓ 1: CLASSIFICANDO ===")
    messages = [
        SystemMessage(content="Classify the user question into exactly one word: TECHNICAL or GENERAL."),
        HumanMessage(content=state["user_input"])
    ]
    result = llm.invoke(messages)
    category = result.content.strip().upper()
    category = "TECHNICAL" if "TECHNICAL" in category else "GENERAL"

    print("Categoria detectada:", category)
    return {"category": category}

def respond(state: State) -> State:
    print("\n=== NÓ 2: RESPONDENDO ===")
    print("Usando categoria:", state["category"])
    messages = [
        SystemMessage(content=f"You are a helpful assistant. The question was classified as {state['category']}. Answer accordingly in Portuguese."),
        HumanMessage(content=state["user_input"])
    ]
    result = llm.invoke(messages)
    print("Resposta:", result.content)
    return {"response": result.content}

def summarize(state: State) -> State:
    print("\n=== NÓ 3: SUMARIZANDO ===")
    print("Resposta:", state["response"])
    messages = [
        SystemMessage(content="Summarize the following response in one sentence."),
        HumanMessage(content=state["response"])
    ]
    result = llm.invoke(messages)
    return {"summary": result.content}

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("respond", respond)
builder.add_node("summarize", summarize)
builder.set_entry_point("classify")
builder.add_edge("classify", "respond")
builder.add_edge("respond", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()

user_input = input("Digite sua pergunta: ")
output = graph.invoke({"user_input": user_input, "category": "", "response": "", "summary": ""})

print("\n=== RESUMO FINAL ===")
print(output["summary"])
