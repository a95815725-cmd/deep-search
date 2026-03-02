"""
Judge prompt loader — reads system + human prompts from JSON files.

Each JSON file contains both messages for one judge, keeping them
together as a logical unit:
  { "system": "...", "human": "..." }

Usage:
    from src.evaluation.prompts import load_judge_prompt
    system, human = load_judge_prompt("planner_judge")
"""

import json
from pathlib import Path
from typing import Tuple

_PROMPTS_DIR = Path(__file__).parent


def load_judge_prompt(name: str) -> Tuple[str, str]:
    """
    Load system and human prompts for a judge by name (without .json extension).

    Args:
        name: Filename stem, e.g. "planner_judge", "reflector_judge"

    Returns:
        Tuple of (system_prompt, human_prompt).

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        KeyError: If the JSON is missing "system" or "human" keys.
    """
    path = _PROMPTS_DIR / f"{name}.json"
    if not path.exists():
        available = [p.stem for p in _PROMPTS_DIR.glob("*.json")]
        raise FileNotFoundError(f"Judge prompt not found: {path}. Available: {available}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["system"], data["human"]
