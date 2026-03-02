import logging
from typing import Dict, List

from pydantic import BaseModel, Field

from src.agent.constants import _SYNTHESIZER_BASE_CHARS, context_budget
from src.agent.prompts import load_prompt
from src.agent.state import AgentState, Citation
from src.models.model_registry import get_llm

logger = logging.getLogger(__name__)

_MAX_SYNTHESIS_DOCS = 20  # max documents fed to synthesizer before score-based truncation
_MAX_CHUNK_CHARS = 1_000  # characters per chunk

SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_HUMAN_PROMPT = load_prompt("synthesizer")


class CitationItem(BaseModel):
    doc_name: str = Field(description="Name/filename of the source document")
    page_num: int = Field(description="Page number in the source document (0 if unknown)", ge=0)
    section: str = Field(
        description="Section of the document (e.g., 'Balance Sheet', 'Risk Factors', 'MD&A')"
    )
    text_excerpt: str = Field(
        description="Brief verbatim excerpt from the document that supports the citation (max 200 chars)"
    )


class SynthesizerOutput(BaseModel):
    answer: str = Field(
        description=(
            "The comprehensive, well-structured financial answer. Must reference retrieved "
            "documents using [Doc N] notation for all factual claims."
        )
    )
    citations: List[CitationItem] = Field(
        description="All citations extracted from the answer, mapped to source documents.",
        default_factory=list,
    )
    confidence_score: float = Field(
        description="Confidence score from 0.0 (very low) to 1.0 (very high).",
        ge=0.0,
        le=1.0,
    )
    remaining_uncertainties: str = Field(
        default="",
        description=(
            "Brief description of any data that could not be found or verified. "
            "Empty string if the answer is fully grounded."
        ),
    )


def _format_context_for_synthesis(retrieved_documents: List[Dict], max_chars: int) -> str:
    """
    Format ranked documents into a numbered context block for the synthesizer prompt.
    Truncates to `max_chars` — callers pass a model-scaled budget via context_budget().
    """
    if not retrieved_documents:
        return "(No documents retrieved.)"

    ranked = sorted(
        retrieved_documents,
        key=lambda d: (0 if d.get("source") == "vector_store" else 1, -(d.get("score") or 0.0)),
    )[:_MAX_SYNTHESIS_DOCS]
    lines: List[str] = []
    total_chars = 0

    for i, doc in enumerate(ranked):
        meta = doc.get("metadata", {})
        source = meta.get("source", "unknown")
        doc_name = meta.get("filename", meta.get("doc_name", source))
        page = meta.get("page", meta.get("page_num", "?"))
        section = meta.get("section", meta.get("header", ""))
        score_str = f"{doc['score']:.3f}" if doc.get("score") is not None else "n/a"
        web_tag = " [WEB]" if doc.get("source") == "web" else ""

        content = doc.get("content", "")[:_MAX_CHUNK_CHARS]
        entry = (
            f"[Doc {i + 1}]{web_tag} {doc_name} | Page {page}"
            + (f" | {section}" if section else "")
            + f" | relevance={score_str}\n{content}\n"
        )

        total_chars += len(entry)
        if total_chars > max_chars:
            remaining = len(ranked) - i
            lines.append(f"\n... [{remaining} additional documents omitted for brevity] ...")
            break
        lines.append(entry)

    return "\n".join(lines)


def synthesizer_node(state: AgentState, config: dict) -> dict:
    """
    Produces the final comprehensive answer grounded in retrieved documents.

    Steps:
      1. Rank and truncate retrieved documents by relevance.
      2. Format context and call LLM with structured output.
      3. Adjust confidence score using objective agent-run signals.
      4. Return final_answer, citations, confidence_score.
    """
    model_name = state["model_name"]
    iteration_count = state.get("iteration_count", 0)
    logger.info("Synthesizer: generating final answer after %d iterations.", iteration_count)

    retrieved_documents = state.get("retrieved_documents", [])
    # Scale context budget to the model's context window (e.g. Gemini 1M → 4× budget).
    max_chars = context_budget(model_name, _SYNTHESIZER_BASE_CHARS)
    context_str = _format_context_for_synthesis(retrieved_documents, max_chars=max_chars)

    gaps = state.get("gaps_identified", [])
    gaps_str = "\n".join(f"- {g}" for g in gaps) if gaps else "None identified."

    try:
        result: SynthesizerOutput = (
            get_llm(model_name)
            .with_structured_output(SynthesizerOutput)
            .invoke(
                [
                    ("system", SYNTHESIZER_SYSTEM_PROMPT),
                    (
                        "human",
                        SYNTHESIZER_HUMAN_PROMPT.format(
                            question=state["original_question"],
                            reflection_notes=state.get("reflection_notes", "N/A"),
                            gaps=gaps_str,
                            context=context_str,
                        ),
                    ),
                ]
            )
        )
    except Exception as exc:
        logger.error("Synthesizer LLM call failed: %s", exc)
        result = SynthesizerOutput(
            answer=(
                f"An error occurred during synthesis: {exc}\n\n"
                "The retrieved documents could not be processed into a final answer. "
                "Please review the reasoning trace for details."
            ),
            citations=[],
            confidence_score=0.0,
            remaining_uncertainties="Synthesis failed due to LLM error.",
        )

    citations: List[Citation] = [
        Citation(
            doc_name=c.doc_name,
            page_num=c.page_num,
            section=c.section,
            text_excerpt=c.text_excerpt[:200],
        )
        for c in result.citations
    ]

    confidence = round(max(0.0, min(1.0, result.confidence_score)), 3)

    uncertainty_note = (
        f" | Uncertainties: {result.remaining_uncertainties}"
        if result.remaining_uncertainties
        else ""
    )
    trace_entry = (
        f"Synthesizer: {len(result.answer)} chars, "
        f"{len(citations)} citations, confidence={confidence:.3f}{uncertainty_note}"
    )
    logger.info(trace_entry)

    return {
        "final_answer": result.answer,
        "citations": citations,
        "confidence_score": confidence,
        "reasoning_trace": [trace_entry],
    }
