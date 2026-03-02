"""
Shared fixtures and helpers for the test suite.

Design principles:
- No real API calls — LLM and embedding calls are always mocked.
- No real ChromaDB on disk — vector store tests use in-memory or mocked clients.
- No real PDF files — parser tests use synthetic text.
"""

import pytest

from src.agent.state import AgentState, Citation, SubQuery


# ---------------------------------------------------------------------------
# Shared AgentState factory
# ---------------------------------------------------------------------------


def make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState, with optional field overrides."""
    base: AgentState = {
        "original_question": "What is the CET1 ratio trend?",
        "model_name": "gpt-4o",
        "sub_queries": [],
        "search_strategy": "",
        "retrieved_documents": [],
        "iteration_count": 0,
        "reflection_notes": "",
        "gaps_identified": [],
        "sufficient_context": False,
        "final_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "reasoning_trace": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def base_state() -> AgentState:
    return make_state()


@pytest.fixture
def state_with_docs() -> AgentState:
    """State that already has some retrieved documents."""
    return make_state(
        retrieved_documents=[
            {
                "content": "The CET1 ratio stood at 14.2% in 2023.",
                "metadata": {"source": "rabobank_2023.pdf", "page": 42, "section": "capital_adequacy"},
                "source": "vector_store",
                "score": 0.91,
            },
            {
                "content": "Total risk-weighted assets were EUR 220 billion.",
                "metadata": {"source": "rabobank_2023.pdf", "page": 44, "section": "capital_adequacy"},
                "source": "vector_store",
                "score": 0.87,
            },
        ],
        sub_queries=[
            SubQuery(query="CET1 capital ratio 2023", status="searched", results=[]),
            SubQuery(query="risk-weighted assets trend", status="searched", results=[]),
        ],
        iteration_count=1,
    )


@pytest.fixture
def sample_sub_queries():
    return [
        SubQuery(query="CET1 ratio 2023", status="pending", results=[]),
        SubQuery(query="CET1 ratio 2022", status="pending", results=[]),
        SubQuery(query="risk-weighted assets", status="pending", results=[]),
    ]
