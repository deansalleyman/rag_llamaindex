"""Multi-agent workflow that directs a query to the right agent.

A single ``AgentWorkflow`` with three agents:

    * ``rag_agent`` (root) -- answers document/knowledge questions by running a
      vector-retrieval (RAG) query over the ChromaDB store, and hands demo
      arithmetic off to the calculation agents.
    * ``add_agent`` / ``multiply_agent`` -- demo calculations.

The RAG query engine is reused from ``rag_pipeline.py`` so vector-store setup
lives in exactly one place. Building the query engine is heavy (loads the embed
model, needs ``HF_TOKEN``), so it happens lazily inside ``build_workflow`` --
importing this module stays cheap and side-effect free.

Run:
    python multiAgentWorkflow.py --query "What is RAG?"
    python multiAgentWorkflow.py --query "What is 5 plus 3, then multiplied by 2?"

Ingest documents first with ``python rag_pipeline.py --reingest``.
"""

import argparse
import asyncio

from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context

from rag_pipeline import build_vector_store, get_query_engine
from resilient_hf_llm import ResilientHuggingFaceInferenceAPI

LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"


# Define some tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # update our count
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a + b


async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # update our count
    cur_state = await ctx.store.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.store.set("state", cur_state)

    return a * b


# Resilient variant: the stock HuggingFaceInferenceAPI closes its shared async
# client after each streaming call, which breaks a multi-step ReAct workflow that
# streams many times on one shared llm. See resilient_hf_llm.py.
llm = ResilientHuggingFaceInferenceAPI(model_name=LLM_MODEL_NAME)

# we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
    can_handoff_to=["rag_agent", "add_agent"],
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
    can_handoff_to=["rag_agent", "multiply_agent"],
)


def build_workflow():
    """Assemble the RAG-routing multi-agent workflow.

    The ``rag_agent`` is the root: every query enters through it. It answers
    document/knowledge questions itself via the RAG query engine and hands demo
    arithmetic off to ``add_agent`` / ``multiply_agent``.

    Building the vector store / query engine happens here (not at import) so it
    is only paid for when the workflow is actually built.
    """
    embed_model, vector_store, chroma_collection = build_vector_store()

    if chroma_collection.count() == 0:
        print(
            "Vector store is empty -- the RAG agent has nothing to retrieve. "
            "Ingest documents first with: python rag_pipeline.py --reingest"
        )

    query_engine = get_query_engine(embed_model, vector_store)
    rag_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="rag_search",
        description=(
            "Answer questions about the indexed document knowledge base using "
            "vector retrieval (RAG)."
        ),
    )

    rag_agent = ReActAgent(
        name="rag_agent",
        description=(
            "Answers questions about the indexed documents using vector "
            "retrieval (RAG)."
        ),
        system_prompt=(
            "A helpful assistant that answers knowledge questions using the "
            "rag_search tool. For arithmetic, hand off to add_agent or "
            "multiply_agent."
        ),
        tools=[rag_tool],
        llm=llm,
        can_handoff_to=["add_agent", "multiply_agent"],
    )

    return AgentWorkflow(
        agents=[rag_agent, addition_agent, multiply_agent],
        root_agent="rag_agent",
        initial_state={"num_fn_calls": 0},
        state_prompt="Current state: {state}. User message: {msg}",
    )


async def run_query(query: str):
    """Build the workflow, run one query, and report the answer + calc calls."""
    workflow = build_workflow()
    ctx = Context(workflow)
    response = await workflow.run(user_msg=query, ctx=ctx)
    state = await ctx.store.get("state")
    print(f"\nQ: {query}\nA: {response}")
    print(f"(calculation tool calls: {state['num_fn_calls']})")
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Route a query through the RAG multi-agent workflow."
    )
    parser.add_argument(
        "--query",
        default="What is RAG?",
        help="Question to route through the workflow.",
    )
    args = parser.parse_args()
    asyncio.run(run_query(args.query))


if __name__ == "__main__":
    main()
