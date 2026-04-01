# graph_07.py — Classify → Chat with Persistent Memory
#
# LEARNING GOALS:
#   - Combine classification with persistent memory in a two-node pipeline
#   - Understand why we classify only the LAST message (not the full history)
#   - See how category metadata influences the response in the chat node
#
# NEW CONCEPT:
#   - Smarter classification: state["messages"][-1] extracts only the most
#     recent message. Sending the full history to a classifier is wasteful
#     and can confuse the model — we only need to classify what's new.
#
# PIPELINE:
#   [user input] → classify (last msg only) → chat (full history) → END
#
# Memory persists across turns via MemorySaver + thread_id (same as graph_06).

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

# --- STATE ---
# Two fields: category (a string, overwritten each turn) and messages
# (an accumulating list, never overwritten — same pattern as graph_06).
class State(TypedDict):
    category: str                   # reset every turn by classify node
    messages: Annotated[list, add]  # grows across turns via MemorySaver

# --- NODE 1: classify ---
# IMPORTANT: We classify only the last message, NOT the full history.
# Reasons:
#   1. Efficiency — the classifier doesn't need context from previous turns
#   2. Accuracy — classifying a dialogue as QUESTION/STATEMENT is ambiguous;
#      classifying just the new message is clearer
def classify(state: State) -> State:
    last_message = state["messages"][-1].content  # grab only the newest message
    messages = [
        SystemMessage(content="Classify the user question into exactly one word: QUESTION or STATEMENT."),
        HumanMessage(content=last_message)
    ]
    result = llm.invoke(messages)
    category = result.content.strip().upper()
    # Safety fallback: if the model added extra words, we still extract correctly
    category = "QUESTION" if "QUESTION" in category else "STATEMENT"

    print("Categoria detectada:", category)
    return {"category": category}

# --- NODE 2: chat ---
# Uses the category from node 1 to adapt its system prompt,
# and sends the full message history to Claude for context-aware replies.
def chat(state: State) -> State:
    print(f"\nHistórico tem {len(state['messages'])} mensagens")

    # The category is injected into the system prompt so Claude knows
    # whether to answer a question or acknowledge a statement.
    all_messages = [
        SystemMessage(content=f"You are a helpful assistant. The question was classified as {state['category']}. Answer accordingly in Portuguese.")
    ] + state["messages"]  # full history included for context

    result = llm.invoke(all_messages)
    print(f"Claude: {result.content}")

    # Offer to display history AFTER responding (avoids cluttering the flow)
    ver_historico = input("Ver histórico? (s/n): ")
    if ver_historico.lower() == "s":
        print("--- Histórico ---")
        for msg in state["messages"]:
            print(f"{type(msg).__name__}: {msg.content}")
        print("----------------")

    # Return only the new AI reply — Annotated[list, add] handles appending
    return {"messages": [result]}

# --- GRAPH ASSEMBLY ---
# Linear chain: classify → chat → END
builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("chat", chat)

builder.set_entry_point("classify")
builder.add_edge("classify", "chat")  # always run chat after classify
builder.add_edge("chat", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- CHAT LOOP ---
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
