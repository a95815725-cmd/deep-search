import logging
from typing import List

from pydantic import BaseModel, Field

from src.agent.prompts import load_prompt
from src.agent.state import AgentState, SubQuery
from src.models.model_registry import get_llm

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT, PLANNER_HUMAN_PROMPT = load_prompt("planner")


class PlannerOutput(BaseModel):
    search_strategy: str = Field(
        description=(
            "A 1-3 sentence description of the overall search approach: what sections of "
            "financial documents to target, what time periods are relevant, and any special "
            "considerations (e.g., need for cross-document comparison)."
        )
    )
    sub_queries: List[str] = Field(
        description=(
            "List of 3-5 targeted sub-queries. Each should be a specific, self-contained "
            "question that will retrieve relevant chunks from financial documents."
        ),
        min_length=3,
        max_length=5,
    )


def planner_node(state: AgentState, config: dict) -> dict:
    """
    Decomposes the original financial question into targeted sub-queries.
    Uses the LLM with structured output to produce a PlannerOutput.
    """
    model_name = state["model_name"]
    logger.info(
        "Planner: using model %s for question: %.80s...", model_name, state["original_question"]
    )

    try:
        result: PlannerOutput = (
            get_llm(model_name)
            .with_structured_output(PlannerOutput)
            .invoke(
                [
                    ("system", PLANNER_SYSTEM_PROMPT),
                    ("human", PLANNER_HUMAN_PROMPT.format(question=state["original_question"])),
                ]
            )
        )
    except Exception as exc:
        logger.error("Planner LLM call failed: %s", exc)
        # Bypass PlannerOutput (min_length=3 would reject a single-query fallback)
        fallback_sq = SubQuery(query=state["original_question"].strip(), status="pending", results=[])
        trace = f"Planner: fallback to direct search due to error: {exc}"
        logger.info(trace)
        return {
            "sub_queries": [fallback_sq],
            "search_strategy": "Direct search on the original question due to planning failure.",
            "reasoning_trace": [trace],
        }

    sub_queries: List[SubQuery] = [
        SubQuery(query=q.strip(), status="pending", results=[])
        for q in result.sub_queries
        if q.strip()
    ]

    query_summaries = "; ".join(f'"{sq["query"]}"' for sq in sub_queries)
    trace_entry = (
        f"Planner: decomposed into {len(sub_queries)} sub-queries: [{query_summaries}] | "
        f"Strategy: {result.search_strategy}"
    )
    logger.info(trace_entry)

    return {
        "sub_queries": sub_queries,
        "search_strategy": result.search_strategy,
        "reasoning_trace": [trace_entry],
    }
