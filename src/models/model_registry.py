"""
Model registry — maps model IDs to LangChain chat model instances.
Supports OpenAI, Anthropic, and Google providers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    # OpenAI
    "gpt-4o": {
        "provider": "openai",
        "display_name": "GPT-4o",
        "context_window": 128_000,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "display_name": "GPT-4o Mini",
        "context_window": 128_000,
    },
    # Anthropic
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "display_name": "Claude 3.5 Sonnet",
        "context_window": 200_000,
    },
    "claude-3-7-sonnet-20250219": {
        "provider": "anthropic",
        "display_name": "Claude 3.7 Sonnet",
        "context_window": 200_000,
    },
    # Google
    "gemini-2.0-flash": {
        "provider": "google",
        "display_name": "Gemini 2.0 Flash",
        "context_window": 1_000_000,
    },
    "gemini-2.0-flash-thinking-exp": {
        "provider": "google",
        "display_name": "Gemini 2.0 Flash Thinking",
        "context_window": 1_000_000,
    },
}

_KEY_MAP = {
    "openai": lambda: settings.OPENAI_API_KEY,
    "anthropic": lambda: settings.ANTHROPIC_API_KEY,
    "google": lambda: settings.GOOGLE_API_KEY,
}


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def get_llm(model_name: str, temperature: float = 0.1, **kwargs: Any):
    """Return a configured LangChain chat model for the given model ID."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {sorted(AVAILABLE_MODELS)}")

    provider = AVAILABLE_MODELS[model_name]["provider"]
    api_key = _KEY_MAP[provider]()

    if not api_key:
        raise EnvironmentError(f"API key for {provider!r} is not set. Add it to your .env file.")

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name, temperature=temperature, openai_api_key=api_key, **kwargs
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name, temperature=temperature, anthropic_api_key=api_key, **kwargs
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature, google_api_key=api_key, **kwargs
        )

    raise ValueError(f"Unsupported provider: {provider!r}")


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Return metadata for a model, including whether its API key is configured."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}")
    info = dict(AVAILABLE_MODELS[model_name])
    info["model_id"] = model_name
    info["api_key_configured"] = bool(_KEY_MAP[info["provider"]]())
    return info


def list_available_models(configured_only: bool = False) -> List[Dict[str, Any]]:
    """List all models. If configured_only=True, only return models with API keys set."""
    models = [get_model_info(m) for m in AVAILABLE_MODELS]
    if configured_only:
        models = [m for m in models if m["api_key_configured"]]
    models.sort(key=lambda m: (not m["api_key_configured"], m["display_name"]))
    return models


def is_model_available(model_name: str) -> bool:
    """True if model is registered and its API key is configured."""
    return model_name in AVAILABLE_MODELS and bool(
        _KEY_MAP[AVAILABLE_MODELS[model_name]["provider"]]()
    )
