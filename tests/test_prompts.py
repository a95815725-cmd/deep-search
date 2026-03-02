"""
Unit tests for prompt loaders.

Tests cover:
  - src/agent/prompts: load_prompt returns (system, human) for all 3 nodes
  - src/evaluation/prompts: load_judge_prompt returns (system, human) for all 3 judges
  - Both raise FileNotFoundError for unknown names
  - Both return non-empty strings
"""

import pytest

from src.agent.prompts import load_prompt
from src.evaluation.prompts import load_judge_prompt


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------


class TestLoadAgentPrompt:
    @pytest.mark.parametrize("name", ["planner", "reflector", "synthesizer"])
    def test_returns_tuple_of_two_strings(self, name):
        result = load_prompt(name)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize("name", ["planner", "reflector", "synthesizer"])
    def test_system_prompt_non_empty(self, name):
        system, _ = load_prompt(name)
        assert isinstance(system, str)
        assert len(system) > 50

    @pytest.mark.parametrize("name", ["planner", "reflector", "synthesizer"])
    def test_human_prompt_non_empty(self, name):
        _, human = load_prompt(name)
        assert isinstance(human, str)
        assert len(human) > 10

    @pytest.mark.parametrize("name", ["planner", "reflector", "synthesizer"])
    def test_human_prompt_has_format_placeholder(self, name):
        _, human = load_prompt(name)
        # Every human prompt has at least one {placeholder} for .format() calls
        assert "{" in human and "}" in human

    def test_planner_human_has_question_placeholder(self):
        _, human = load_prompt("planner")
        assert "{question}" in human

    def test_reflector_human_has_all_placeholders(self):
        _, human = load_prompt("reflector")
        for key in ["{question}", "{search_strategy}", "{iteration}", "{context}", "{sub_queries_summary}"]:
            assert key in human, f"Missing placeholder: {key}"

    def test_synthesizer_human_has_all_placeholders(self):
        _, human = load_prompt("synthesizer")
        for key in ["{question}", "{reflection_notes}", "{gaps}", "{context}"]:
            assert key in human, f"Missing placeholder: {key}"

    def test_unknown_name_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_node")


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------


class TestLoadJudgePrompt:
    @pytest.mark.parametrize("name", ["planner_judge", "reflector_judge", "synthesizer_judge"])
    def test_returns_tuple_of_two_strings(self, name):
        result = load_judge_prompt(name)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize("name", ["planner_judge", "reflector_judge", "synthesizer_judge"])
    def test_both_strings_non_empty(self, name):
        system, human = load_judge_prompt(name)
        assert len(system) > 20
        assert len(human) > 10

    def test_unknown_judge_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_judge_prompt("nonexistent_judge")
