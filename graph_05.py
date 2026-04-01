# graph_05.py — Three-Node Pipeline: Classify → Respond → Summarize
#
# LEARNING GOALS:
#   - Build a multi-node graph where each node feeds the next
#   - Use classification output to change a downstream node's behavior
#   - See how state grows as it passes through a pipeline
#
# NEW CONCEPTS:
#   - Multi-step LLM chaining: three separate API calls, each with a purpose
#   - Conditional prompting: node 2 reads the category set by node 1
#   - Terminal input: reading the user's question at runtime with input()
#
# PIPELINE:
#   [user] → classify → respond → summarize → [print summary]
#
# State at each stage:
#   after classify:  category is set
#   after respond:   response is set (using category)
#   after summarize: summary is set (using response)

import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)

# --- STATE ---
# More fields than before — each node contributes its own piece.
class State(TypedDict):
    user_input: str   # the raw question from the user
    category: str     # "TECHNICAL" or "GENERAL" — set by classify node
    response: str     # the full answer — set by respond node
    summary: str      # one-sentence condensed answer — set by summarize node

# --- NODE 1: classify ---
# Asks the LLM to label the question with one word.
# Note: we use .strip().upper() to normalize the output (LLMs can be inconsistent),
# then a simple "in" check as a safety fallback in case the model adds extra words.
def classify(state: State) -> State:
    print("=== NÓ 1: CLASSIFICANDO ===")
    messages = [
        SystemMessage(content="Classify the user question into exactly one word: TECHNICAL or GENERAL."),
        HumanMessage(content=state["user_input"])
    ]
    result = llm.invoke(messages)
    category = result.content.strip().upper()
    # Fallback: if the model wrote anything other than "TECHNICAL", treat as GENERAL
    category = "TECHNICAL" if "TECHNICAL" in category else "GENERAL"

    print("Categoria detectada:", category)
    return {"category": category}  # only update this one field

# --- NODE 2: respond ---
# Uses the category from the previous node to adapt the system prompt.
# This is a simple example of conditional behavior based on upstream state.
def respond(state: State) -> State:
    print("\n=== NÓ 2: RESPONDENDO ===")
    print("Usando categoria:", state["category"])
    messages = [
        # The f-string embeds the category into the system prompt dynamically
        SystemMessage(content=f"You are a helpful assistant. The question was classified as {state['category']}. Answer accordingly in Portuguese."),
        HumanMessage(content=state["user_input"])
    ]
    result = llm.invoke(messages)
    print("Resposta:", result.content)
    return {"response": result.content}

# --- NODE 3: summarize ---
# Takes the full response from node 2 and distills it into one sentence.
# This shows how later nodes can process the outputs of earlier ones.
def summarize(state: State) -> State:
    print("\n=== NÓ 3: SUMARIZANDO ===")
    print("Resposta:", state["response"])
    messages = [
        SystemMessage(content="Summarize the following response in one sentence."),
        HumanMessage(content=state["response"])  # passes the previous node's output
    ]
    result = llm.invoke(messages)
    return {"summary": result.content}

# --- GRAPH ASSEMBLY ---
# A linear chain: each node connects to the next with a simple edge.
builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("respond", respond)
builder.add_node("summarize", summarize)

builder.set_entry_point("classify")
builder.add_edge("classify", "respond")    # classify → respond
builder.add_edge("respond", "summarize")   # respond → summarize
builder.add_edge("summarize", END)

graph = builder.compile()

# --- RUN ---
# input() pauses execution and waits for the user to type a question.
user_input = input("Digite sua pergunta: ")
output = graph.invoke({"user_input": user_input, "category": "", "response": "", "summary": ""})

print("\n=== RESUMO FINAL ===")
print(output["summary"])
