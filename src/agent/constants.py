# Agent-wide constants shared across nodes and the graph router.

MAX_ITERATIONS = 5  # Maximum search-reflect cycles before forcing synthesis

# Context budget baseline — calibrated for GPT-4o (128k window).
# Larger-window models (Claude 200k, Gemini 1M) get proportionally more context.
_BASE_WINDOW = 128_000
_REFLECTOR_BASE_CHARS = 12_000  # reflector needs a snapshot, not full detail
_SYNTHESIZER_BASE_CHARS = 18_000  # synthesizer needs richer context for citations


def context_budget(model_name: str, base_chars: int) -> int:
    """
    Scale a context character budget proportionally to the model's context window.
    Caps at 4× the baseline to avoid diminishing returns from very large windows.

    Args:
        model_name: Registered model ID (e.g. "gpt-4o", "gemini-2.0-flash").
        base_chars: Budget calibrated for the 128k baseline model.

    Returns:
        Adjusted character budget for this model.
    """
    from src.models.model_registry import AVAILABLE_MODELS

    window = AVAILABLE_MODELS.get(model_name, {}).get("context_window", _BASE_WINDOW)
    scale = min(window / _BASE_WINDOW, 4.0)
    return int(base_chars * scale)
