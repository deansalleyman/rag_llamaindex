# RAG-routing multi-agent workflow — Design

**Date:** 2026-07-24
**Status:** Approved (design), pending implementation
**Scope:** Extend `multiAgentWorkflow.py` so a multi-agent workflow directs an
incoming query — knowledge questions are answered by a RAG agent (vector
retrieval over ChromaDB), while demo arithmetic is handled by dedicated
calculation agents.

## Goal

Add the ability to route a single natural-language query through a multi-agent
method. The workflow:

- Sends document/knowledge questions to a **RAG agent** that returns the
  synthesized RAG answer (vector-retrieval-backed, LLM-composed).
- Hands off arithmetic to the existing **calculation agents** (`add`,
  `multiply`).

This is the LlamaIndex `AgentWorkflow` handoff pattern, wiring the existing math
agents in `multiAgentWorkflow.py` together with a new RAG agent that wraps the
query engine already defined in `rag_pipeline.py`.

## Non-goals (YAGNI)

- No changes to ingestion or the vector store schema. Ingestion stays in
  `rag_pipeline.py`.
- The RAG agent returns the **synthesized answer only** — not raw retrieved
  nodes, similarity scores, or source filenames. (Explicitly chosen.)
- No new LLM/embedding stack. Reuse HuggingFace `bge-small-en-v1.5` embeddings +
  `Qwen/Qwen2.5-Coder-32B-Instruct` via the HF Inference API, matching
  `rag_pipeline.py`.
- No web UI / Gradio integration. Command-line entry point only.

## Architecture

A single `AgentWorkflow` with three agents:

| Agent | Role | Tool(s) | Handoffs |
|-------|------|---------|----------|
| `rag_agent` (**root**) | Directs the query. Answers document/knowledge questions itself via RAG. Hands arithmetic to the calc agents. | `QueryEngineTool` wrapping `rag_pipeline.get_query_engine()` | `add_agent`, `multiply_agent` |
| `add_agent` | Demo calculation | existing `add` | `rag_agent`, `multiply_agent` |
| `multiply_agent` | Demo calculation | existing `multiply` | `rag_agent`, `add_agent` |

`rag_agent` is `root_agent`, so every query enters through it. It answers
knowledge questions directly; for arithmetic it hands off. Calc agents can hand
back to `rag_agent` (and to each other) so a mixed query can be completed.

### Data flow

```
user_msg
   │
   ▼
rag_agent (root)  ──answers knowledge Qs──▶  QueryEngineTool ──▶ ChromaDB vector store
   │                                                             (bge-small embeddings)
   │                                                                    │
   │                                              retrieved chunks ──▶ Qwen LLM (tree_summarize)
   │                                                                    │
   │◀──────────────────── synthesized answer ───────────────────────────┘
   │
   └──arithmetic──▶ add_agent / multiply_agent ──(add/multiply tool, updates num_fn_calls)──▶ result
```

## Components

### Reuse from `rag_pipeline.py`
- `build_vector_store()` → `(embed_model, vector_store, chroma_collection)`
- `get_query_engine(embed_model, vector_store)` → query engine
  (`response_mode="tree_summarize"`, raises a clear error if `HF_TOKEN` is
  unset).

No Chroma/embedding/LLM setup is duplicated in `multiAgentWorkflow.py`; it is
imported from `rag_pipeline.py`, keeping that module the single source of truth.

### `multiAgentWorkflow.py` (extended)
- **Keep** `add(ctx, a, b)` and `multiply(ctx, a, b)`, including the
  `num_fn_calls` state updates via `ctx.store`.
- **Keep** `add_agent` and `multiply_agent`, adding `can_handoff_to` so they can
  route to `rag_agent` and each other.
- **Add** `build_workflow()`:
  - Builds the vector store and query engine (lazy — inside the function, not at
    import time, so importing the module stays cheap and side-effect free).
  - Wraps the query engine in a `QueryEngineTool` named e.g. `rag_search` with a
    description like *"Answer questions about the indexed document knowledge base
    using vector retrieval (RAG)."*
  - Constructs `rag_agent` (`ReActAgent`) with that tool and
    `can_handoff_to=["add_agent", "multiply_agent"]`.
  - Returns an `AgentWorkflow(agents=[rag_agent, add_agent, multiply_agent],
    root_agent="rag_agent", initial_state={"num_fn_calls": 0},
    state_prompt="Current state: {state}. User message: {msg}")`.
- **Add** an `async def run_query(query: str)` helper that builds the workflow,
  creates a `Context`, runs it, prints the answer and the final
  `num_fn_calls`.
- **Add** a `__main__` block with `argparse` (`--query`, default sample query)
  that runs `run_query` via `asyncio.run`.

The module-level `llm` (shared `HuggingFaceInferenceAPI`) is retained for the
calc agents; the RAG query engine uses the token-aware LLM built inside
`get_query_engine`.

## Implementation note: streaming-safe HF LLM

Discovered during implementation. The stock `HuggingFaceInferenceAPI` closes its
shared async client at the end of every streaming call
(`llama_index/llms/huggingface_api/base.py`, `astream_chat` / `astream_complete`),
but the client is created once in `__init__`. A ReAct multi-agent workflow
streams many times on one shared LLM, so after the first stream every later call
raises `RuntimeError: Cannot send a request, as the client has been closed`.

`resilient_hf_llm.py` provides `ResilientHuggingFaceInferenceAPI`, a thin
subclass that refreshes the async client before each streaming call. The shared
agent `llm` in `multiAgentWorkflow.py` uses it. Covered by
`test_resilient_hf_llm.py` (offline, patched clients). Assumes streams on a given
instance are sequential (true for a single-query ReAct handoff flow).

## Error handling

- **Missing `HF_TOKEN`:** `get_query_engine` already raises a `RuntimeError`
  with guidance to set it in `.env`. `build_workflow()` lets it propagate.
- **Empty vector store:** if `chroma_collection.count() == 0`, print a warning
  that no documents are ingested and point the user to
  `python rag_pipeline.py --reingest`, then continue (the RAG agent will simply
  have nothing to retrieve).

## Testing

A lightweight test (no network calls) that:

1. Calls `build_workflow()` and asserts it returns an `AgentWorkflow` with
   exactly three agents.
2. Asserts `root_agent` is `rag_agent`.
3. Asserts the handoff wiring: `rag_agent.can_handoff_to` includes both calc
   agents.

To keep the test fully offline and fast, it **mocks
`rag_pipeline.get_query_engine`** (and `build_vector_store`) to return stubs, so
`build_workflow()` assembles the agents without loading the embed model,
requiring `HF_TOKEN`, or making any network/inference call. No inference is
performed in the test.

## Run

```bash
# Ensure documents are ingested first (one-time / on change):
python rag_pipeline.py --reingest

# Route a knowledge query through the multi-agent workflow:
python multiAgentWorkflow.py --query "What is RAG?"

# Route a demo calculation:
python multiAgentWorkflow.py --query "What is 5 plus 3, then multiplied by 2?"
```

Output prints the workflow's answer and the final `num_fn_calls`, so calc
handoffs are observable.
