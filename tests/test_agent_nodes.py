"""
Unit tests for agent nodes — all LLM and vector store calls are mocked.

Tests cover:
  - planner_node: structured output, fallback on LLM failure
  - searcher_node: parallel search, deduplication, status update
  - reflector_node: sufficient/insufficient routing, tool result clearing, dedup queries
  - synthesizer_node: citations, confidence clamping, fallback on failure
  - should_continue router: all three routing outcomes
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agent.graph import should_continue
from src.agent.planner import planner_node
from src.agent.reflector import reflector_node
from src.agent.searcher import searcher_node
from src.agent.state import SubQuery
from src.agent.synthesizer import synthesizer_node
from tests.conftest import make_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm(return_value):
    """Return a mock LLM chain that yields `return_value` from .invoke()."""
    llm = MagicMock()
    chain = MagicMock()
    chain.invoke.return_value = return_value
    llm.with_structured_output.return_value = chain
    return llm


def _mock_vector_store(results=None):
    """Return a mock vector store whose .search() returns `results`."""
    vs = MagicMock()
    vs.search.return_value = results or []
    return vs


# ---------------------------------------------------------------------------
# planner_node
# ---------------------------------------------------------------------------


class TestPlannerNode:
    def _run(self, question="What is the CET1 trend?", sub_queries=None, strategy="direct search"):
        from src.agent.planner import PlannerOutput

        llm_output = PlannerOutput(
            search_strategy=strategy,
            sub_queries=sub_queries or ["CET1 ratio 2023", "CET1 ratio 2022", "risk-weighted assets"],
        )
        state = make_state(original_question=question)
        config = {}

        with patch("src.agent.planner.get_llm", return_value=_mock_llm(llm_output)):
            return planner_node(state, config)

    def test_returns_sub_queries_as_pending(self):
        result = self._run()
        for sq in result["sub_queries"]:
            assert sq["status"] == "pending"
            assert sq["results"] == []

    def test_returns_correct_number_of_sub_queries(self):
        result = self._run(sub_queries=["q1", "q2", "q3"])
        assert len(result["sub_queries"]) == 3

    def test_search_strategy_in_result(self):
        result = self._run(strategy="temporal decomposition by year")
        assert result["search_strategy"] == "temporal decomposition by year"

    def test_reasoning_trace_appended(self):
        result = self._run()
        assert len(result["reasoning_trace"]) == 1
        assert "Planner" in result["reasoning_trace"][0]

    def test_fallback_on_llm_failure(self):
        state = make_state(original_question="What is the CET1 trend?")
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("API down")

        with patch("src.agent.planner.get_llm", return_value=llm):
            result = planner_node(state, {})

        assert len(result["sub_queries"]) == 1
        assert result["sub_queries"][0]["query"] == "What is the CET1 trend?"

    def test_blank_sub_queries_filtered(self):
        from src.agent.planner import PlannerOutput

        llm_output = PlannerOutput(
            search_strategy="test",
            sub_queries=["valid query", "  ", "another valid"],
        )
        state = make_state()
        with patch("src.agent.planner.get_llm", return_value=_mock_llm(llm_output)):
            result = planner_node(state, {})
        queries = [sq["query"] for sq in result["sub_queries"]]
        assert "" not in queries
        assert "  " not in queries


# ---------------------------------------------------------------------------
# searcher_node
# ---------------------------------------------------------------------------


class TestSearcherNode:
    def _raw_doc(self, content, page=1):
        return {"text": content, "metadata": {"source": "doc.pdf", "page": page}, "score": 0.9}

    def test_pending_queries_become_searched(self):
        state = make_state(
            sub_queries=[
                SubQuery(query="CET1 ratio", status="pending", results=[]),
                SubQuery(query="risk-weighted assets", status="pending", results=[]),
            ]
        )
        vs = _mock_vector_store([self._raw_doc("CET1 was 14.2%")])
        config = {"configurable": {"vector_store": vs}}

        result = searcher_node(state, config)
        for sq in result["sub_queries"]:
            assert sq["status"] == "searched"

    def test_already_searched_queries_unchanged(self):
        state = make_state(
            sub_queries=[
                SubQuery(query="old query", status="searched", results=[]),
                SubQuery(query="new query", status="pending", results=[]),
            ]
        )
        vs = _mock_vector_store([])
        config = {"configurable": {"vector_store": vs}}
        result = searcher_node(state, config)

        statuses = {sq["query"]: sq["status"] for sq in result["sub_queries"]}
        assert statuses["old query"] == "searched"
        assert statuses["new query"] == "searched"

    def test_iteration_count_incremented(self):
        state = make_state(
            iteration_count=1,
            sub_queries=[SubQuery(query="q", status="pending", results=[])],
        )
        config = {"configurable": {"vector_store": _mock_vector_store([])}}
        result = searcher_node(state, config)
        assert result["iteration_count"] == 2

    def test_duplicate_docs_not_added_twice(self):
        doc = self._raw_doc("The CET1 ratio was 14.2%", page=5)
        state = make_state(
            retrieved_documents=[
                {
                    "content": doc["text"],
                    "metadata": doc["metadata"],
                    "source": "vector_store",
                    "score": 0.9,
                }
            ],
            sub_queries=[SubQuery(query="CET1", status="pending", results=[])],
        )
        vs = _mock_vector_store([doc])  # same doc returned again
        config = {"configurable": {"vector_store": vs}}
        result = searcher_node(state, config)
        # The doc was already in state — should NOT be in the new docs list
        assert result["retrieved_documents"] == []

    def test_new_unique_docs_returned(self):
        state = make_state(
            sub_queries=[SubQuery(query="CET1", status="pending", results=[])],
        )
        doc = self._raw_doc("New unique content about CET1 capital requirements")
        vs = _mock_vector_store([doc])
        config = {"configurable": {"vector_store": vs}}
        result = searcher_node(state, config)
        assert len(result["retrieved_documents"]) == 1

    def test_reasoning_trace_contains_iter_info(self):
        state = make_state(
            sub_queries=[SubQuery(query="q", status="pending", results=[])],
        )
        config = {"configurable": {"vector_store": _mock_vector_store([])}}
        result = searcher_node(state, config)
        assert "Searcher" in result["reasoning_trace"][0]


# ---------------------------------------------------------------------------
# reflector_node
# ---------------------------------------------------------------------------


class TestReflectorNode:
    def _run(self, sufficient=True, gaps=None, follow_up=None, state_overrides=None):
        from src.agent.reflector import ReflectionOutput

        llm_output = ReflectionOutput(
            reflection_notes="Analysis complete.",
            gaps_identified=gaps or [],
            sufficient_context=sufficient,
            follow_up_queries=follow_up or [],
        )
        state = make_state(**(state_overrides or {}))
        with patch("src.agent.reflector.get_llm", return_value=_mock_llm(llm_output)):
            return reflector_node(state, {})

    def test_sufficient_context_set_true(self):
        result = self._run(sufficient=True)
        assert result["sufficient_context"] is True

    def test_insufficient_context_set_false(self):
        result = self._run(sufficient=False)
        assert result["sufficient_context"] is False

    def test_gaps_propagated(self):
        gaps = ["Missing CET1 for 2022", "No balance sheet data"]
        result = self._run(sufficient=False, gaps=gaps)
        assert result["gaps_identified"] == gaps

    def test_follow_up_queries_added_as_pending(self):
        result = self._run(
            sufficient=False,
            follow_up=["CET1 ratio 2022", "balance sheet 2022"],
        )
        new_pending = [sq for sq in result["sub_queries"] if sq["status"] == "pending"]
        assert len(new_pending) == 2

    def test_tool_result_clearing(self):
        # SubQuery.results should be cleared after reflector pass
        state = make_state(
            sub_queries=[
                SubQuery(query="q1", status="searched", results=[{"content": "some doc"}]),
            ]
        )
        from src.agent.reflector import ReflectionOutput

        llm_output = ReflectionOutput(
            reflection_notes="ok",
            gaps_identified=[],
            sufficient_context=True,
            follow_up_queries=[],
        )
        with patch("src.agent.reflector.get_llm", return_value=_mock_llm(llm_output)):
            result = reflector_node(state, {})
        for sq in result["sub_queries"]:
            assert sq["results"] == []

    def test_duplicate_follow_up_queries_deduplicated(self):
        existing = make_state(
            sub_queries=[SubQuery(query="cet1 ratio 2023", status="searched", results=[])],
        )
        from src.agent.reflector import ReflectionOutput

        llm_output = ReflectionOutput(
            reflection_notes="gaps",
            gaps_identified=["Missing 2022"],
            sufficient_context=False,
            follow_up_queries=["CET1 ratio 2023"],  # exact duplicate of existing
        )
        with patch("src.agent.reflector.get_llm", return_value=_mock_llm(llm_output)):
            result = reflector_node(existing, {})
        new_pending = [sq for sq in result["sub_queries"] if sq["status"] == "pending"]
        assert len(new_pending) == 0  # duplicate was rejected

    def test_reasoning_trace_appended(self):
        result = self._run(sufficient=True)
        assert len(result["reasoning_trace"]) == 1
        assert "Reflector" in result["reasoning_trace"][0]


# ---------------------------------------------------------------------------
# synthesizer_node
# ---------------------------------------------------------------------------


class TestSynthesizerNode:
    def _run(self, answer="Good answer.", citations=None, confidence=0.85, state_overrides=None):
        from src.agent.synthesizer import CitationItem, SynthesizerOutput

        cit_items = []
        for c in (citations or []):
            cit_items.append(CitationItem(
                doc_name=c.get("doc_name", "doc.pdf"),
                page_num=c.get("page_num", 1),
                section=c.get("section", "notes"),
                text_excerpt=c.get("text_excerpt", "excerpt"),
            ))

        llm_output = SynthesizerOutput(
            answer=answer,
            citations=cit_items,
            confidence_score=confidence,
            remaining_uncertainties="",
        )
        state = make_state(**(state_overrides or {}))
        with patch("src.agent.synthesizer.get_llm", return_value=_mock_llm(llm_output)):
            return synthesizer_node(state, {})

    def test_final_answer_set(self):
        result = self._run(answer="The CET1 ratio trended upward from 13.5% to 14.2%.")
        assert result["final_answer"] == "The CET1 ratio trended upward from 13.5% to 14.2%."

    def test_citations_returned(self):
        result = self._run(citations=[{"doc_name": "report.pdf", "page_num": 42,
                                        "section": "capital_adequacy", "text_excerpt": "CET1 14.2%"}])
        assert len(result["citations"]) == 1
        assert result["citations"][0]["doc_name"] == "report.pdf"

    def test_confidence_score_within_valid_range(self):
        # Pydantic enforces ge=0.0, le=1.0 on SynthesizerOutput.
        # The synthesizer then rounds the value to 3 decimal places.
        result = self._run(confidence=0.856789)
        assert 0.0 <= result["confidence_score"] <= 1.0
        # Should be rounded to 3 decimal places
        assert result["confidence_score"] == round(0.856789, 3)

    def test_confidence_score_at_boundaries(self):
        result_min = self._run(confidence=0.0)
        assert result_min["confidence_score"] == 0.0

        result_max = self._run(confidence=1.0)
        assert result_max["confidence_score"] == 1.0

    def test_reasoning_trace_appended(self):
        result = self._run()
        assert len(result["reasoning_trace"]) == 1
        assert "Synthesizer" in result["reasoning_trace"][0]

    def test_fallback_on_llm_failure(self):
        llm = MagicMock()
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("API timeout")
        state = make_state()
        with patch("src.agent.synthesizer.get_llm", return_value=llm):
            result = synthesizer_node(state, {})
        assert "error" in result["final_answer"].lower()
        assert result["confidence_score"] == 0.0


# ---------------------------------------------------------------------------
# should_continue router
# ---------------------------------------------------------------------------


class TestShouldContinue:
    def test_sufficient_context_routes_to_synthesize(self):
        state = make_state(sufficient_context=True, iteration_count=1)
        assert should_continue(state) == "synthesize"

    def test_iteration_cap_forces_synthesize(self):
        state = make_state(sufficient_context=False, iteration_count=5)
        assert should_continue(state) == "synthesize"

    def test_gaps_within_cap_routes_to_search(self):
        state = make_state(sufficient_context=False, iteration_count=2)
        assert should_continue(state) == "search"

    def test_iteration_exactly_at_cap_routes_to_synthesize(self):
        from src.agent.constants import MAX_ITERATIONS

        state = make_state(sufficient_context=False, iteration_count=MAX_ITERATIONS)
        assert should_continue(state) == "synthesize"

    def test_zero_iterations_routes_to_search(self):
        state = make_state(sufficient_context=False, iteration_count=0)
        assert should_continue(state) == "search"
