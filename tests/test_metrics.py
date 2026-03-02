"""
Unit tests for src/evaluation/metrics.py

Tests cover:
  - detect_hallucinations: happy path, rounding tolerance, empty inputs, all verified
  - calculate_answer_completeness: full match, partial, zero, empty hints
"""

from src.evaluation.metrics import calculate_answer_completeness, detect_hallucinations

# ---------------------------------------------------------------------------
# detect_hallucinations
# ---------------------------------------------------------------------------


class TestDetectHallucinations:
    def _doc(self, content: str) -> dict:
        return {"content": content, "metadata": {}}

    def test_all_numbers_verified(self):
        answer = "The CET1 ratio was 14.2% and RWA was 220 billion."
        docs = [self._doc("CET1 ratio 14.2% and RWA 220 billion in 2023.")]
        result = detect_hallucinations(answer, docs)
        assert result["hallucination_rate"] == 0.0
        assert "14.2" in result["verified_numbers"]
        assert "220" in result["verified_numbers"]

    def test_hallucinated_number_flagged(self):
        answer = "Revenue grew to 999 billion."
        docs = [self._doc("Revenue was 100 billion last year.")]
        result = detect_hallucinations(answer, docs)
        assert "999" in result["hallucinated_numbers"]
        assert result["hallucination_rate"] > 0

    def test_rounding_tolerance_accepted(self):
        # 14.2 rounds to 14.18 — within 1% tolerance
        answer = "CET1 ratio is 14.2%."
        docs = [self._doc("CET1 ratio stood at 14.18 percent.")]
        result = detect_hallucinations(answer, docs)
        assert "14.2" in result["verified_numbers"]
        assert result["hallucination_rate"] == 0.0

    def test_empty_answer_returns_zeros(self):
        result = detect_hallucinations("", [self._doc("some text 100")])
        assert result["hallucination_rate"] == 0.0
        assert result["total_answer_numbers"] == 0

    def test_no_docs_all_numbers_hallucinated(self):
        answer = "Revenue was 500 million."
        result = detect_hallucinations(answer, [])
        assert "500" in result["hallucinated_numbers"]
        assert result["hallucination_rate"] == 1.0

    def test_numbers_0_and_1_ignored(self):
        # 0 and 1 are too common to be meaningful hallucination signals
        answer = "Tier 1 capital ratio improved by 0 basis points."
        docs = [self._doc("No matching numbers here at all.")]
        result = detect_hallucinations(answer, docs)
        # 0 and 1 should not appear in answer_numbers at all
        assert "0" not in result["hallucinated_numbers"]
        assert "1" not in result["hallucinated_numbers"]

    def test_partial_hallucination_rate(self):
        answer = "CET1 was 14.2% and Tier 2 was 999%."
        docs = [self._doc("CET1 14.2 percent.")]
        result = detect_hallucinations(answer, docs)
        assert "14.2" in result["verified_numbers"]
        assert "999" in result["hallucinated_numbers"]
        assert 0 < result["hallucination_rate"] < 1


# ---------------------------------------------------------------------------
# calculate_answer_completeness
# ---------------------------------------------------------------------------


class TestCalculateAnswerCompleteness:
    def test_all_hints_present(self):
        answer = "The CET1 ratio was 14.2%, with risk-weighted assets at 220 billion."
        hints = ["CET1 ratio", "risk-weighted assets"]
        assert calculate_answer_completeness(answer, hints) == 1.0

    def test_partial_hints_present(self):
        answer = "The CET1 ratio was 14.2%."
        hints = ["CET1 ratio", "risk-weighted assets", "Basel III"]
        score = calculate_answer_completeness(answer, hints)
        assert score == round(1 / 3, 4)

    def test_no_hints_present(self):
        answer = "Revenue grew this quarter."
        hints = ["CET1 ratio", "risk-weighted assets"]
        assert calculate_answer_completeness(answer, hints) == 0.0

    def test_empty_hints_returns_1(self):
        # No hints = nothing to check = full score by convention
        assert calculate_answer_completeness("any answer", []) == 1.0

    def test_empty_answer_returns_0(self):
        assert calculate_answer_completeness("", ["CET1 ratio"]) == 0.0

    def test_case_insensitive(self):
        answer = "cet1 RATIO WAS 14.2%"
        hints = ["CET1 ratio"]
        assert calculate_answer_completeness(answer, hints) == 1.0

    def test_single_hint_present(self):
        answer = "The Basel III framework requires minimum capital buffers."
        hints = ["Basel III"]
        assert calculate_answer_completeness(answer, hints) == 1.0
