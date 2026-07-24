"""Offline tests for the RAG-routing multi-agent workflow.

These tests mock the RAG setup (``build_vector_store`` / ``get_query_engine``)
so ``build_workflow`` assembles the agents without loading the embed model,
requiring HF_TOKEN, or making any network/inference call.
"""

from unittest.mock import MagicMock, patch


def test_build_workflow_routes_rag_and_calc_agents():
    """build_workflow() wires a rag_agent root that hands off to the calc agents."""
    with patch(
        "multiAgentWorkflow.build_vector_store",
        return_value=(MagicMock(), MagicMock(), MagicMock()),
    ), patch(
        "multiAgentWorkflow.get_query_engine",
        return_value=MagicMock(),
    ):
        from multiAgentWorkflow import build_workflow

        workflow = build_workflow()

    # Exactly the three expected agents.
    assert set(workflow.agents.keys()) == {"rag_agent", "add_agent", "multiply_agent"}

    # The RAG agent directs the query (it is the entry point).
    assert workflow.root_agent == "rag_agent"

    # The RAG agent can hand demo calculations off to both calc agents.
    rag_handoffs = workflow.agents["rag_agent"].can_handoff_to
    assert "add_agent" in rag_handoffs
    assert "multiply_agent" in rag_handoffs
