# LangGraph Studio — Learning Examples

A progressive series of LangGraph examples built with Python, exploring state graphs from basics to multi-node LLM pipelines powered by Claude (Anthropic).

## Requirements

- Python 3.11+
- An `ANTHROPIC_API_KEY` environment variable set

```bash
pip install -r requirements.txt
```

## Examples

### graph_01.py — Basic StateGraph (no LLM)
Introduces the core LangGraph concepts: **State**, **Nodes**, and **Edges**.

- Defines a `State` TypedDict with `message` and `result` fields.
- Two nodes (`node_a`, `node_b`) process the state sequentially.
- No LLM involved — pure Python functions chained together.
- Good starting point to understand how data flows through a graph.

### graph_02.py — Single Node with Claude
Integrates a real LLM (Claude Haiku) into a LangGraph graph.

- One node calls `ChatAnthropic` with the user's input.
- The graph invokes Claude and returns the response in the state.
- Demonstrates the minimal setup needed to use an LLM inside a graph.

### graph_03.py — Inspecting the LLM Response Object
Extends `graph_02` with detailed debug output to explore LangChain's response structure.

- Prints the full `AIMessage` object returned by the LLM, including `content`, `response_metadata`, and `id`.
- Useful for understanding what data is available after an LLM call.

### graph_04.py — System + Human Messages
Introduces structured message formatting using `SystemMessage` and `HumanMessage`.

- Sets up a system prompt instructing Claude to respond in Portuguese.
- Shows how to pass a list of messages (instead of a plain string) to the LLM.
- Demonstrates prompt engineering within a graph node.

### graph_05.py — Three-Node Pipeline: Classify → Respond → Summarize
A multi-node graph that chains three LLM calls together, with terminal input.

- **Input**: reads the user's question at runtime via `input()` from the terminal.
- **Node 1 (`classify`)**: asks Claude to classify the question as `TECHNICAL` or `GENERAL`, with robust parsing (`.strip().upper()` + fallback to `GENERAL`).
- **Node 2 (`respond`)**: uses the detected category to tailor the response, answering in Portuguese.
- **Node 3 (`summarize`)**: condenses the full response into a single summary sentence.
- State now carries a `summary` field in addition to `category` and `response`.
- Illustrates multi-step LLM chaining where each node's output feeds the next.

## Concepts Covered

| Concept | Introduced in |
|---|---|
| `StateGraph`, `TypedDict` state | graph_01 |
| Nodes and edges | graph_01 |
| LLM integration (`ChatAnthropic`) | graph_02 |
| LangChain response object inspection | graph_03 |
| `SystemMessage` / `HumanMessage` | graph_04 |
| Multi-node chaining | graph_05 |
| State passing between nodes | graph_05 |
| Terminal user input (`input()`) | graph_05 |
| Three-step LLM pipeline | graph_05 |
