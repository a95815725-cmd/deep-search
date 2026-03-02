"""
Agent prompt loader — reads system + human prompts from JSON files.

Each JSON file contains both messages for one node, keeping them
together as a logical unit:
  { "system": "...", "human": "..." }

Usage:
    from src.agent.prompts import load_prompt
    system, human = load_prompt("planner")
"""

import json
from pathlib import Path
from typing import Tuple

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> Tuple[str, str]:
    """
    Load system and human prompts for an agent node by name (without .json extension).

    Args:
        name: Filename stem, e.g. "planner", "reflector", "synthesizer"

    Returns:
        Tuple of (system_prompt, human_prompt).

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        KeyError: If the JSON is missing "system" or "human" keys.
    """
    path = _PROMPTS_DIR / f"{name}.json"
    if not path.exists():
        available = [p.stem for p in _PROMPTS_DIR.glob("*.json")]
        raise FileNotFoundError(f"Agent prompt not found: {path}. Available: {available}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["system"], data["human"]
