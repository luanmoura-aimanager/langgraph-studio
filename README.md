# LangGraph Studio — Learning Examples

A progressive series of LangGraph examples built with Python. Each file introduces one new concept, building on the previous one — from a basic stateful pipeline to a branching, memory-enabled chat with an LLM.

By the end of the series you'll understand how to design multi-node LLM pipelines, persist conversation history, and route execution dynamically based on model output.

## Requirements

- Python 3.11+
- An `ANTHROPIC_API_KEY` environment variable set

```bash
pip install -r requirements.txt
```

---

## How to read these examples

Each file is self-contained and runnable. Start from `graph_01.py` and work your way up. The comments inside each file explain *why* each piece exists, not just *what* it does.

---

## Examples

### graph_01.py — Basic StateGraph (no LLM)

**What it teaches:** The three core building blocks of every LangGraph program.

- **State** — a `TypedDict` that acts as a shared "notebook" passed between all nodes. Every node reads from it and writes updates back to it.
- **Nodes** — plain Python functions. They receive the current state and return a dict with only the fields they want to update.
- **Edges** — define the execution order. `add_edge("a", "b")` means "after node a finishes, run node b". `END` stops the graph.

No LLM involved — this is the cleanest way to understand data flow before adding AI complexity.

---

### graph_02.py — Single Node with Claude

**What it teaches:** The minimal setup to call an LLM inside a graph node.

- Introduces `ChatAnthropic`, LangChain's wrapper around the Anthropic API.
- The node calls `llm.invoke(text)` and extracts the response with `.content`.
- Shows that adding an LLM to a graph is just a function call inside a node — the graph structure stays the same.

---

### graph_03.py — Inspecting the LLM Response Object

**What it teaches:** What the LLM actually returns (it's more than just text).

- `llm.invoke()` returns an `AIMessage` object, not a plain string.
- `.content` — the text response you usually care about.
- `.response_metadata` — token usage, stop reason, model name (useful for cost tracking and debugging).
- `.id` — a unique ID for this specific API call.
- Also inspects the `state` object and the compiled `graph` object so you know their types.

> Tip: run this file and study the printed output carefully — understanding the response shape pays off in later examples.

---

### graph_04.py — System + Human Messages

**What it teaches:** How to structure messages for a chat model.

- Chat models expect a **list of message objects**, not a plain string.
- `SystemMessage` — sets the assistant's persona, rules, and constraints (e.g. "always reply in Portuguese").
- `HumanMessage` — carries the user's actual question.
- Order matters: system message first, then human message.
- The model reads the whole list and responds accordingly.

This is the foundation for all prompt engineering in LangGraph.

---

### graph_05.py — Three-Node Pipeline: Classify → Respond → Summarize

**What it teaches:** Multi-step LLM chaining — each node's output becomes the next node's input.

- **Node 1 (`classify`)**: asks Claude to label the question as `TECHNICAL` or `GENERAL`. Uses `.strip().upper()` + a string `in` check to handle inconsistent model output.
- **Node 2 (`respond`)**: reads the category from state and embeds it in the system prompt, adapting the response style.
- **Node 3 (`summarize`)**: takes the full response from node 2 and condenses it to one sentence.
- The state gains a new field at each step: `category` → `response` → `summary`.
- Also introduces `input()` for reading the user's question at runtime.

---

### graph_06.py — Persistent Memory Chat

**What it teaches:** How to keep conversation history across multiple turns.

- **`MemorySaver`** — a built-in LangGraph checkpointer. Pass it to `compile()` and LangGraph automatically saves and loads state between invocations.
- **`Annotated[list, add]`** — tells LangGraph to *append* new messages to the list instead of overwriting it. This is the standard pattern for accumulating message history.
- **`thread_id`** — a string key in the `config` dict that isolates one user's history from another's. Same name → same history loaded.

On every turn, the node sees the *full conversation history*, not just the latest message.

---

### graph_07.py — Classify → Chat with Persistent Memory

**What it teaches:** Combining classification with persistent memory in a two-node pipeline.

- **Smarter classification**: `state["messages"][-1]` extracts only the most recent message to classify. Sending the full history to a classifier is wasteful and can confuse the model.
- **`category` in state**: a string field (not a list) that gets overwritten each turn, while `messages` keeps accumulating.
- History is displayed on demand *after* the response, keeping the flow clean.

This file combines everything from graph_05 and graph_06.

---

### graph_08.py — Conditional Edges (Branching)

**What it teaches:** How to route execution to different nodes at runtime based on state.

Until now every edge was fixed — node A always went to node B. **Conditional edges** let the graph *decide* which node to run next.

- **`add_conditional_edges(source, router_fn, mapping)`**:
  1. The source node runs and updates state.
  2. `router_fn(state)` is called — it returns a string (the "route key").
  3. `mapping[route_key]` gives the name of the next node to execute.
- The router function is a plain Python function — not a node. Keep it simple: just read state and return a string.
- Two response nodes (`respond_technical`, `respond_general`) share the same graph but use different system prompts depending on the classification.
- Memory persists across turns via `MemorySaver` + `thread_id`.

```
              ┌── "TECHNICAL" ──→ respond_technical ──→ END
   classify ──┤
              └── "GENERAL"   ──→ respond_general   ──→ END
```

---

## Concepts Covered

| Concept | Introduced in |
|---|---|
| `StateGraph`, `TypedDict` state | graph_01 |
| Nodes and edges | graph_01 |
| `END` sentinel | graph_01 |
| LLM integration (`ChatAnthropic`) | graph_02 |
| `AIMessage` response object | graph_03 |
| `response_metadata`, token usage | graph_03 |
| `SystemMessage` / `HumanMessage` | graph_04 |
| Structured message lists | graph_04 |
| Multi-node chaining | graph_05 |
| Conditional prompting via state | graph_05 |
| Terminal user input (`input()`) | graph_05 |
| `MemorySaver` / persistent memory | graph_06 |
| `Annotated[list, add]` accumulation | graph_06 |
| Per-user `thread_id` context | graph_06 |
| Classify only the last message | graph_07 |
| Classify + Chat pipeline with memory | graph_07 |
| Conditional edges (branching) | graph_08 |
| Router functions | graph_08 |
| Multiple response paths in one graph | graph_08 |
