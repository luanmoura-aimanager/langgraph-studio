from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. STATE — o "caderno" compartilhado entre todos os nós
class State(TypedDict):
    message: str
    result: str

# 2. NODES — funções que recebem o state e retornam um state atualizado
def node_a(state: State) -> State:
    print("Node A executando...")
    return {"result": f"Node A processou: '{state['message']}'"}

def node_b(state: State) -> State:
    print("Node B executando...")
    return {"result": state["result"] + " → Node B também passou aqui"}

# 3. GRAPH — conecta tudo
builder = StateGraph(State)

builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)

builder.set_entry_point("node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

graph = builder.compile()

# 4. RODAR
output = graph.invoke({"message": "olá LangGraph", "result": ""})
print("\nEstado final:", output)
