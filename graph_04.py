# graph_04.py — System + Human Messages
#
# LEARNING GOALS:
#   - Understand the difference between SystemMessage and HumanMessage
#   - Learn how to structure a list of messages to send to an LLM
#   - See how a system prompt shapes the model's behavior
#
# NEW CONCEPT: Instead of sending a plain string to the LLM, we send a list
# of structured message objects. This is how chat models expect input:
#
#   SystemMessage  →  sets the assistant's persona, rules, and context
#   HumanMessage   →  the actual user question or request
#
# The model reads the whole list and responds accordingly.

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
    response: str

# --- NODE ---
def call_claude(state: State) -> State:
    # Build the message list that the LLM will receive.
    # Order matters: SystemMessage first, then HumanMessage.
    messages = [
        # SystemMessage instructs the model HOW to behave.
        # Here we ask it to always reply in Portuguese.
        SystemMessage(content="You are a helpful assistant that answers in Portuguese."),
        # HumanMessage carries the user's actual question from the state.
        HumanMessage(content=state["user_input"])
    ]

    # Print the messages before sending — useful for debugging prompt structure
    print("=== MENSAGENS ENVIADAS ===")
    for msg in messages:
        print(f"{type(msg).__name__}: {msg.content}")
    print()

    # Pass the list of messages (not just a string) to the LLM
    result = llm.invoke(messages)

    print("=== RESPOSTA ===")
    print(result.content)

    return {"response": result.content}

# --- GRAPH ASSEMBLY ---
builder = StateGraph(State)
builder.add_node("call_claude", call_claude)
builder.set_entry_point("call_claude")
builder.add_edge("call_claude", END)

graph = builder.compile()

# --- RUN ---
# Even though the question is in English, Claude will answer in Portuguese
# because of the SystemMessage instruction.
output = graph.invoke({"user_input": "What is LangGraph in one sentence?", "response": ""})
