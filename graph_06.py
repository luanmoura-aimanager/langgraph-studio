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

class State(TypedDict):
    messages: Annotated[list, add]

def chat(state: State) -> State:
    print(f"\nHistórico tem {len(state['messages'])} mensagens")

    all_messages = [
        SystemMessage(content="You are a helpful assistant. Answer in Portuguese.")
    ] + state["messages"]

    print("--- Histórico ---")
    ver_historico = input("Ver histórico? (s/n): ")
    if ver_historico.lower() == "s":
        for msg in state["messages"]:
            print(f"{type(msg).__name__}: {msg.content}")
    print("----------------")
    
    result = llm.invoke(all_messages)
    print(f"Claude: {result.content}")
    return {"messages": [result]}

builder = StateGraph(State)
builder.add_node("chat", chat)
builder.set_entry_point("chat")
builder.add_edge("chat", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

print("Chat com memória! Digite 'sair' para encerrar.\n")
while True:
    thread = input("Nome do usuário: ")
    config = {"configurable": {"thread_id": thread.lower()}}

    user_input = input("Digite sua mensagem: ")
    if user_input.lower() == "sair":
        break
    
    graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
