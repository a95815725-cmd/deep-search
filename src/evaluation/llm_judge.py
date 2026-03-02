"""
LLM-as-Judge evaluation for each agent node.

Three independent judges run concurrently via asyncio.gather:
  - judge_planner     : was the question decomposition good?
  - judge_reflector   : was the gap assessment accurate?
  - judge_synthesizer : is the final answer faithful to the source docs?

Usage:
    import asyncio
    from src.evaluation.llm_judge import run_all_judges

    scores = asyncio.run(run_all_judges(agent_result))
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List  # Any kept for run_all_judges output dict

from pydantic import BaseModel, Field

from src.evaluation.prompts import load_judge_prompt
from src.models.model_registry import get_llm

logger = logging.getLogger(__name__)

# Load all judge prompts at module init — fails fast if a file is missing
_PLANNER_SYSTEM, _PLANNER_HUMAN = load_judge_prompt("planner_judge")
_REFLECTOR_SYSTEM, _REFLECTOR_HUMAN = load_judge_prompt("reflector_judge")
_SYNTHESIZER_SYSTEM, _SYNTHESIZER_HUMAN = load_judge_prompt("synthesizer_judge")


# ---------------------------------------------------------------------------
# Shared output schema
# ---------------------------------------------------------------------------


class JudgeOutput(BaseModel):
    score: int = Field(
        description="Quality score: 1=very poor, 2=poor, 3=acceptable, 4=good, 5=excellent",
        ge=1,
        le=5,
    )
    verdict: str = Field(
        description="'pass' if score >= 4, 'partial' if score == 3, 'fail' if score <= 2",
    )
    reasoning: str = Field(
        description="2-4 sentence explanation justifying the score.",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific problems found. Empty list if verdict is 'pass'.",
    )


# ---------------------------------------------------------------------------
# Judge 1 — Planner
# ---------------------------------------------------------------------------


async def judge_planner(
    question: str,
    sub_queries: List[str],
    search_strategy: str,
    model_name: str = "gpt-4o-mini",
) -> JudgeOutput:
    """Judge whether the planner decomposed the question effectively."""
    sq_text = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(sub_queries))
    return (
        await get_llm(model_name, temperature=0.0)
        .with_structured_output(JudgeOutput)
        .ainvoke(
            [
                ("system", _PLANNER_SYSTEM),
                (
                    "human",
                    _PLANNER_HUMAN.format(
                        question=question,
                        search_strategy=search_strategy,
                        sub_queries=sq_text,
                    ),
                ),
            ]
        )
    )


# ---------------------------------------------------------------------------
# Judge 2 — Reflector
# ---------------------------------------------------------------------------


async def judge_reflector(
    question: str,
    reflection_notes: str,
    gaps_identified: List[str],
    sufficient_context: bool,
    model_name: str = "gpt-4o-mini",
) -> JudgeOutput:
    """Judge whether the reflector correctly assessed context sufficiency."""
    gaps_text = (
        "\n".join(f"  - {g}" for g in gaps_identified) if gaps_identified else "  (none identified)"
    )
    return (
        await get_llm(model_name, temperature=0.0)
        .with_structured_output(JudgeOutput)
        .ainvoke(
            [
                ("system", _REFLECTOR_SYSTEM),
                (
                    "human",
                    _REFLECTOR_HUMAN.format(
                        question=question,
                        reflection_notes=reflection_notes,
                        gaps=gaps_text,
                        sufficient="YES — proceeded to synthesis"
                        if sufficient_context
                        else "NO — requested more searches",
                    ),
                ),
            ]
        )
    )


# ---------------------------------------------------------------------------
# Judge 3 — Synthesizer
# ---------------------------------------------------------------------------


async def judge_synthesizer(
    question: str,
    answer: str,
    citations: List[Dict],
    retrieved_docs: List[Dict],
    model_name: str = "gpt-4o-mini",
) -> JudgeOutput:
    """Judge whether the synthesizer produced a faithful, complete answer."""
    top_docs = sorted(retrieved_docs, key=lambda d: -(d.get("score") or 0.0))[:5]
    context_text = (
        "\n\n".join(
            f"[Doc {i + 1}] {doc.get('metadata', {}).get('source', 'unknown')}\n{doc.get('content', '')[:600]}"
            for i, doc in enumerate(top_docs)
        )
        or "(no documents)"
    )

    citations_text = (
        "\n".join(
            f"  - {c.get('doc_name', '?')} p.{c.get('page_num', '?')} "
            f"[{c.get('section', '?')}]: {c.get('text_excerpt', '')[:120]}"
            for c in citations
        )
        if citations
        else "  (no citations)"
    )

    return (
        await get_llm(model_name, temperature=0.0)
        .with_structured_output(JudgeOutput)
        .ainvoke(
            [
                ("system", _SYNTHESIZER_SYSTEM),
                (
                    "human",
                    _SYNTHESIZER_HUMAN.format(
                        question=question,
                        context=context_text,
                        answer=answer[:2_000],
                        citations=citations_text,
                    ),
                ),
            ]
        )
    )


# ---------------------------------------------------------------------------
# Run all judges in parallel
# ---------------------------------------------------------------------------


async def run_all_judges(
    agent_result: Dict,
    judge_model: str = "gpt-4o-mini",
) -> Dict:
    """
    Run all three judges concurrently against a single agent run result.

    Args:
        agent_result: The dict returned by run_deep_search (must include the
                      extended fields: original_question, sub_queries,
                      search_strategy, reflection_notes, retrieved_documents).
        judge_model:  LLM to use for all judges. Defaults to gpt-4o-mini
                      (fast and cheap for evaluation).

    Returns:
        Dict with keys: planner, reflector, synthesizer (each a JudgeOutput),
        and overall_score (float average of the three scores).
    """
    sq_raw = agent_result.get("sub_queries", [])
    sub_query_strings = [sq["query"] if isinstance(sq, dict) else str(sq) for sq in sq_raw]

    results = await asyncio.gather(
        judge_planner(
            question=agent_result["original_question"],
            sub_queries=sub_query_strings,
            search_strategy=agent_result.get("search_strategy", ""),
            model_name=judge_model,
        ),
        judge_reflector(
            question=agent_result["original_question"],
            reflection_notes=agent_result.get("reflection_notes", ""),
            gaps_identified=agent_result.get("gaps_identified", []),
            sufficient_context=agent_result.get("sufficient_context", False),
            model_name=judge_model,
        ),
        judge_synthesizer(
            question=agent_result["original_question"],
            answer=agent_result.get("answer", ""),
            citations=agent_result.get("citations", []),
            retrieved_docs=agent_result.get("retrieved_documents", []),
            model_name=judge_model,
        ),
        return_exceptions=True,
    )

    labels = ("planner", "reflector", "synthesizer")
    output: Dict[str, Any] = {}
    scores = []

    for label, result in zip(labels, results, strict=False):
        if isinstance(result, Exception):
            logger.error("Judge '%s' failed: %s", label, result)
            output[label] = JudgeOutput(
                score=1,
                verdict="fail",
                reasoning=f"Judge failed with error: {result}",
                issues=[str(result)],
            )
        else:
            output[label] = result
            scores.append(result.score)

    output["overall_score"] = round(sum(scores) / len(scores), 2) if scores else 0.0
    return output
