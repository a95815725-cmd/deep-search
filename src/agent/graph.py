"""
LangGraph StateGraph definition for the financial deep-search agent.

Flow:
    START → planner → searcher → reflector ─┬─(gaps & iter<5)→ searcher (loop)
                                             └─(sufficient | iter≥5)→ synthesizer → END
"""

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from src.agent.constants import MAX_ITERATIONS
from src.agent.planner import planner_node
from src.agent.reflector import reflector_node
from src.agent.searcher import searcher_node
from src.agent.state import AgentState
from src.agent.synthesizer import synthesizer_node

logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> str:
    """
    Routing function called after the reflector node.

    Returns:
        "synthesize" — if context is sufficient or the iteration cap has been reached.
        "search"     — if gaps remain and more iterations are allowed.
    """
    sufficient = state.get("sufficient_context", False)
    iteration_count = state.get("iteration_count", 0)

    if sufficient:
        logger.info("Router: sufficient context after %d iterations → synthesize", iteration_count)
        return "synthesize"

    if iteration_count >= MAX_ITERATIONS:
        logger.warning(
            "Router: iteration cap (%d) reached with gaps remaining → synthesize anyway",
            MAX_ITERATIONS,
        )
        return "synthesize"

    logger.info(
        "Router: gaps found at iteration %d/%d → search again",
        iteration_count,
        MAX_ITERATIONS,
    )
    return "search"


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Nodes:
        planner    — Decomposes the question into sub-queries
        searcher   — Retrieves relevant chunks from vector store 
        reflector  — Evaluates context sufficiency and generates follow-up queries
        synthesizer — Produces the final grounded answer with citations

    Edges:
        START → planner → searcher → reflector
        reflector → searcher  (conditional: gaps remain and iter < MAX_ITERATIONS)
        reflector → synthesizer (conditional: sufficient or iter cap reached)
        synthesizer → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("reflector", reflector_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "searcher")
    graph.add_edge("searcher", "reflector")
    graph.add_conditional_edges(
        "reflector",
        should_continue,
        {
            "search": "searcher",
            "synthesize": "synthesizer",
        },
    )
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Module-level compiled graph — import this directly for most use cases.
deep_search_graph = build_graph()


def run_deep_search(
    question: str,
    model_name: str = "gpt-4o",
    vector_store: Optional[Any] = None,
    config_overrides: Optional[Dict] = None,
) -> Dict:
    """
    Run the financial deep-search agent on a single question.

    Args:
        question:        The complex financial question to answer.
        model_name:      LLM model identifier (must be registered in model_registry).
                         Defaults to "gpt-4o".
        vector_store:    An initialised FinancialVectorStore instance.
                         If None, the searcher node will create one with default settings.
        config_overrides: Optional dict of additional LangGraph configurable values.

    Returns:
        A dict containing:
            answer          (str)        — The final synthesised answer.
            citations       (list[dict]) — Extracted citations (doc_name, page, section, excerpt).
            confidence_score (float)     — Confidence from 0.0 to 1.0.
            reasoning_trace (list[str])  — Step-by-step log of agent decisions.
            iteration_count (int)        — Number of search-reflect cycles executed.
            model_name      (str)        — Model used for the run.
            gaps_identified (list[str])  — Any gaps that were not fully resolved.
            sufficient_context (bool)    — Whether the reflector reached satisfaction.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string.")

    initial_state: AgentState = {
        "original_question": question.strip(),
        "model_name": model_name,
        # Planning
        "sub_queries": [],
        "search_strategy": "",
        # Retrieval — Annotated[List, operator.add] fields must start as empty list
        "retrieved_documents": [],
        "iteration_count": 0,
        # Reflection
        "reflection_notes": "",
        "gaps_identified": [],
        "sufficient_context": False,
        # Output
        "final_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "reasoning_trace": [],
    }

    # Build configurable dict for LangGraph
    configurable: Dict = {"vector_store": vector_store}
    if config_overrides:
        configurable.update(config_overrides)

    run_config = {"configurable": configurable}

    logger.info(
        "run_deep_search: starting agent | model=%s | question='%.80s...'",
        model_name,
        question,
    )

    try:
        final_state: AgentState = deep_search_graph.invoke(initial_state, config=run_config)
    except Exception as exc:
        logger.error("Deep search agent run failed: %s", exc, exc_info=True)
        return {
            "answer": f"Agent run failed: {exc}",
            "citations": [],
            "confidence_score": 0.0,
            "reasoning_trace": [f"FATAL ERROR: {exc}"],
            "iteration_count": 0,
            "model_name": model_name,
            "gaps_identified": [],
            "sufficient_context": False,
        }

    return {
        # Core output
        "answer": final_state.get("final_answer", ""),
        "citations": final_state.get("citations", []),
        "confidence_score": final_state.get("confidence_score", 0.0),
        "reasoning_trace": final_state.get("reasoning_trace", []),
        "iteration_count": final_state.get("iteration_count", 0),
        "model_name": model_name,
        # Extended fields — used by LLM-as-judge evaluation
        "original_question": question.strip(),
        "sub_queries": final_state.get("sub_queries", []),
        "search_strategy": final_state.get("search_strategy", ""),
        "reflection_notes": final_state.get("reflection_notes", ""),
        "gaps_identified": final_state.get("gaps_identified", []),
        "sufficient_context": final_state.get("sufficient_context", False),
        "retrieved_documents": final_state.get("retrieved_documents", []),
    }
