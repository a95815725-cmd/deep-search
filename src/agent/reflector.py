import logging
from typing import Dict, List

from pydantic import BaseModel, Field

from src.agent.constants import _REFLECTOR_BASE_CHARS, MAX_ITERATIONS, context_budget
from src.agent.prompts import load_prompt
from src.agent.state import AgentState, SubQuery
from src.models.model_registry import get_llm

logger = logging.getLogger(__name__)

_MAX_CHUNK_CHARS = 800  # characters per chunk shown to the reflector

REFLECTOR_SYSTEM_PROMPT, REFLECTOR_HUMAN_PROMPT = load_prompt("reflector")


class ReflectionOutput(BaseModel):
    reflection_notes: str = Field(
        description=(
            "A detailed analysis of the quality and completeness of the retrieved context. "
            "Describe what was found, what coverage exists, and any concerns about the data quality."
        )
    )
    gaps_identified: List[str] = Field(
        description=(
            "A list of specific, concrete missing pieces of information. "
            "Each item should be a precise statement like "
            "'Missing: revenue figures for fiscal year 2022' or "
            "'No balance sheet data retrieved for any period'. "
            "Empty list if no meaningful gaps."
        )
    )
    sufficient_context: bool = Field(
        description=(
            "True if the retrieved context is sufficient to produce a well-grounded, "
            "comprehensive answer to the original question. False if important information is missing."
        )
    )
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description=(
            "New, targeted search queries to fill the identified gaps. "
            "Should be different from the sub-queries already executed. "
            "Empty list if sufficient_context is True or no actionable follow-ups exist."
        ),
    )


def _format_context(retrieved_documents: List[Dict], max_chars: int) -> str:
    """
    Format retrieved documents into a compact context string for the reflector prompt.
    Prioritises documents with higher relevance scores.
    Truncates to `max_chars` — callers pass a model-scaled budget via context_budget().
    """
    if not retrieved_documents:
        return "(No documents retrieved yet.)"

    # Sort by score descending (None scores sort last)
    sorted_docs = sorted(
        retrieved_documents,
        key=lambda d: (d.get("score") is None, -(d.get("score") or 0)),
    )

    lines: List[str] = []
    total_chars = 0

    for i, doc in enumerate(sorted_docs):
        meta = doc.get("metadata", {})
        source = meta.get("source", "unknown")
        page = meta.get("page", meta.get("page_num", "?"))
        section = meta.get("section", meta.get("header", ""))
        score_str = f"{doc['score']:.3f}" if doc.get("score") is not None else "n/a"
        content = doc.get("content", "")[:_MAX_CHUNK_CHARS]

        entry = (
            f"[Doc {i + 1}] source={source} page={page}"
            + (f" section={section}" if section else "")
            + f" score={score_str}\n{content}\n"
        )
        total_chars += len(entry)
        if total_chars > max_chars:
            lines.append(f"... ({len(sorted_docs) - i} more documents truncated)")
            break
        lines.append(entry)

    return "\n".join(lines)


def _format_sub_queries_summary(sub_queries: List[SubQuery]) -> str:
    lines = []
    for sq in sub_queries:
        n_results = len(sq.get("results", []))
        lines.append(f"  [{sq['status']}] {sq['query']} → {n_results} results")
    return "\n".join(lines) if lines else "(none)"


def reflector_node(state: AgentState, config: dict) -> dict:
    """
    Evaluates whether the retrieved context is sufficient to answer the original question.

    If gaps are found and the iteration limit has not been reached, generates new pending
    sub-queries to address the missing information.
    """
    model_name = state["model_name"]
    iteration_count = state.get("iteration_count", 0)
    logger.info("Reflector [iter %d]: evaluating context sufficiency.", iteration_count)

    # Scale context budget to the model's context window (larger models get more context).
    max_chars = context_budget(model_name, _REFLECTOR_BASE_CHARS)
    context_str = _format_context(state.get("retrieved_documents", []), max_chars=max_chars)
    sub_queries_summary = _format_sub_queries_summary(state.get("sub_queries", []))

    try:
        result: ReflectionOutput = (
            get_llm(model_name)
            .with_structured_output(ReflectionOutput)
            .invoke(
                [
                    ("system", REFLECTOR_SYSTEM_PROMPT),
                    (
                        "human",
                        REFLECTOR_HUMAN_PROMPT.format(
                            question=state["original_question"],
                            search_strategy=state.get("search_strategy", "N/A"),
                            iteration=iteration_count,
                            context=context_str,
                            sub_queries_summary=sub_queries_summary,
                        ),
                    ),
                ]
            )
        )
    except Exception as exc:
        logger.error("Reflector LLM call failed: %s", exc)
        # Conservative fallback: assume insufficient to avoid silent failures
        result = ReflectionOutput(
            reflection_notes=f"Reflection failed due to LLM error: {exc}",
            gaps_identified=["Reflection could not be completed — treating as insufficient."],
            sufficient_context=False,
            follow_up_queries=[],
        )

    # Build new pending sub-queries from follow-up queries, avoiding duplicates
    existing_queries = {sq["query"].lower().strip() for sq in state.get("sub_queries", [])}
    new_sub_queries: List[SubQuery] = []

    if (
        not result.sufficient_context
        and result.follow_up_queries
        and iteration_count < MAX_ITERATIONS
    ):
        for fq in result.follow_up_queries:
            fq_clean = fq.strip()
            if fq_clean and fq_clean.lower() not in existing_queries:
                new_sub_queries.append(SubQuery(query=fq_clean, status="pending", results=[]))
                existing_queries.add(fq_clean.lower())

    # Tool result clearing:
    # SubQuery.results are raw chunks that have already been promoted to retrieved_documents.
    # Keeping them doubles the context cost on every subsequent reflector pass — clear them here.
    cleared_sub_queries: List[SubQuery] = [
        SubQuery(query=sq["query"], status=sq["status"], results=[])
        for sq in state.get("sub_queries", [])
    ]
    updated_sub_queries = cleared_sub_queries + new_sub_queries

    verdict = (
        "sufficient" if result.sufficient_context else f"gaps found ({len(result.gaps_identified)})"
    )
    trace_entry = (
        f"Reflector [iter {iteration_count}]: {verdict} — "
        f"{result.reflection_notes[:200]}"
        + (
            f" | New follow-up queries: {[q['query'] for q in new_sub_queries]}"
            if new_sub_queries
            else ""
        )
    )
    logger.info(trace_entry)

    return {
        "reflection_notes": result.reflection_notes,
        "gaps_identified": result.gaps_identified,
        "sufficient_context": result.sufficient_context,
        "sub_queries": updated_sub_queries,
        "reasoning_trace": [trace_entry],
    }
