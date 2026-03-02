"""
Unit tests for src/models/model_registry.py and src/agent/constants.py

Tests cover:
  - get_llm raises ValueError for unknown model
  - get_llm raises EnvironmentError when API key missing
  - get_model_info returns correct metadata
  - list_available_models returns all registered models
  - is_model_available returns False without API key
  - context_budget scales proportionally and caps at 4x
"""

from unittest.mock import patch

import pytest

from src.agent.constants import context_budget
from src.models.model_registry import (
    AVAILABLE_MODELS,
    get_model_info,
    is_model_available,
    list_available_models,
)


# ---------------------------------------------------------------------------
# get_llm
# ---------------------------------------------------------------------------


class TestGetLlm:
    def test_unknown_model_raises_value_error(self):
        from src.models.model_registry import get_llm

        with pytest.raises(ValueError, match="Unknown model"):
            get_llm("gpt-99-turbo-ultra")

    def test_missing_api_key_raises_environment_error(self):
        from src.models.model_registry import get_llm

        with patch("src.models.model_registry._KEY_MAP", {"openai": lambda: ""}):
            with pytest.raises(EnvironmentError, match="API key"):
                get_llm("gpt-4o")

    def test_valid_openai_model_returns_llm_instance(self):
        from src.models.model_registry import get_llm

        with patch("src.models.model_registry._KEY_MAP", {"openai": lambda: "sk-fake-key-1234"}):
            llm = get_llm("gpt-4o")
            assert llm is not None

    def test_valid_anthropic_model_returns_llm_instance(self):
        from src.models.model_registry import get_llm

        with patch(
            "src.models.model_registry._KEY_MAP",
            {"anthropic": lambda: "sk-ant-fake-key"},
        ):
            llm = get_llm("claude-3-5-sonnet-20241022")
            assert llm is not None


# ---------------------------------------------------------------------------
# get_model_info
# ---------------------------------------------------------------------------


class TestGetModelInfo:
    def test_gpt4o_info_correct(self):
        info = get_model_info("gpt-4o")
        assert info["provider"] == "openai"
        assert info["context_window"] == 128_000
        assert info["model_id"] == "gpt-4o"

    def test_claude_info_correct(self):
        info = get_model_info("claude-3-5-sonnet-20241022")
        assert info["provider"] == "anthropic"
        assert info["context_window"] == 200_000

    def test_gemini_info_correct(self):
        info = get_model_info("gemini-2.0-flash")
        assert info["provider"] == "google"
        assert info["context_window"] == 1_000_000

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            get_model_info("not-a-model")

    def test_api_key_configured_false_without_key(self):
        with patch("src.models.model_registry._KEY_MAP", {"openai": lambda: ""}):
            info = get_model_info("gpt-4o")
            assert info["api_key_configured"] is False

    def test_api_key_configured_true_with_key(self):
        with patch("src.models.model_registry._KEY_MAP", {"openai": lambda: "sk-fake"}):
            info = get_model_info("gpt-4o")
            assert info["api_key_configured"] is True


# ---------------------------------------------------------------------------
# list_available_models
# ---------------------------------------------------------------------------


class TestListAvailableModels:
    def test_returns_all_registered_models(self):
        models = list_available_models()
        assert len(models) == len(AVAILABLE_MODELS)

    def test_configured_only_filters_correctly(self):
        # No API keys set → configured_only should return empty list
        with patch("src.models.model_registry._KEY_MAP", {p: lambda: "" for p in ["openai", "anthropic", "google"]}):
            models = list_available_models(configured_only=True)
            assert models == []

    def test_each_model_has_required_keys(self):
        models = list_available_models()
        for m in models:
            assert "model_id" in m
            assert "provider" in m
            assert "context_window" in m
            assert "api_key_configured" in m


# ---------------------------------------------------------------------------
# is_model_available
# ---------------------------------------------------------------------------


class TestIsModelAvailable:
    def test_unknown_model_returns_false(self):
        assert is_model_available("gpt-99") is False

    def test_known_model_without_key_returns_false(self):
        with patch("src.models.model_registry._KEY_MAP", {"openai": lambda: ""}):
            assert is_model_available("gpt-4o") is False

    def test_known_model_with_key_returns_true(self):
        with patch("src.models.model_registry._KEY_MAP", {"openai": lambda: "sk-fake"}):
            assert is_model_available("gpt-4o") is True


# ---------------------------------------------------------------------------
# context_budget (from constants.py)
# ---------------------------------------------------------------------------


class TestContextBudget:
    def test_gpt4o_returns_base_chars(self):
        # gpt-4o has 128k window = 1x base
        result = context_budget("gpt-4o", 12_000)
        assert result == 12_000

    def test_claude_scales_proportionally(self):
        # claude-3-5-sonnet has 200k window = 200k/128k ≈ 1.5625x
        result = context_budget("claude-3-5-sonnet-20241022", 12_000)
        expected = int(12_000 * (200_000 / 128_000))
        assert result == expected

    def test_gemini_capped_at_4x(self):
        # gemini has 1M window = 7.8x base, but capped at 4x
        result = context_budget("gemini-2.0-flash", 12_000)
        assert result == 12_000 * 4

    def test_unknown_model_uses_base_window(self):
        # Falls back to _BASE_WINDOW (128k) = 1x
        result = context_budget("unknown-model-xyz", 12_000)
        assert result == 12_000

    def test_synthesizer_budget_larger_than_reflector(self):
        from src.agent.constants import _REFLECTOR_BASE_CHARS, _SYNTHESIZER_BASE_CHARS

        reflector = context_budget("gpt-4o", _REFLECTOR_BASE_CHARS)
        synthesizer = context_budget("gpt-4o", _SYNTHESIZER_BASE_CHARS)
        assert synthesizer > reflector
