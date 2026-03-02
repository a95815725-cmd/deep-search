"""
Central configuration for the Rabo Deep Search agent.
Loads all secrets and tuneable parameters from a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Resolve the project root (directory this file lives in) and load .env
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


class Settings:
    """
    Immutable-by-convention settings object populated from environment
    variables (with sensible defaults).  A single instance is created at
    import time and re-exported as `settings`.
    """

    # ------------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ------------------------------------------------------------------
    # Retrieval / agent loop
    # ------------------------------------------------------------------
    MAX_RETRIEVAL_ITERATIONS: int = 5
    TOP_K_RESULTS: int = 8

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    AVAILABLE_MODELS: List[str] = [
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "gemini-2.0-flash",
    ]
    DEFAULT_MODEL: str = "gpt-4o"

    def __init__(self) -> None:
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        self.CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

        self.MAX_RETRIEVAL_ITERATIONS = int(os.getenv("MAX_RETRIEVAL_ITERATIONS", "5"))
        self.TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "8"))

        self.DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o")
        self.AVAILABLE_MODELS = [
            "gpt-4o",
            "claude-3-5-sonnet-20241022",
            "gemini-2.0-flash",
        ]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_api_key(self, model: str) -> Optional[str]:
        """Return the appropriate API key for a given model name."""
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            return self.OPENAI_API_KEY or None
        if "claude" in model_lower:
            return self.ANTHROPIC_API_KEY or None
        if "gemini" in model_lower:
            return self.GOOGLE_API_KEY or None
        return None

    def validate(self) -> List[str]:
        """
        Return a list of human-readable warnings for missing keys.
        Callers can decide how to surface these (log, raise, display).
        """
        warnings: List[str] = []
        if not self.OPENAI_API_KEY:
            warnings.append(
                "OPENAI_API_KEY is not set — GPT-4o and embeddings will be unavailable."
            )
        if not self.ANTHROPIC_API_KEY:
            warnings.append("ANTHROPIC_API_KEY is not set — Claude models will be unavailable.")
        if not self.GOOGLE_API_KEY:
            warnings.append("GOOGLE_API_KEY is not set — Gemini models will be unavailable.")
        return warnings

    def __repr__(self) -> str:  # pragma: no cover
        def _mask(val: str) -> str:
            if not val:
                return "<not set>"
            return val[:4] + "****" + val[-4:] if len(val) > 8 else "****"

        return (
            f"Settings("
            f"OPENAI_API_KEY={_mask(self.OPENAI_API_KEY)}, "
            f"ANTHROPIC_API_KEY={_mask(self.ANTHROPIC_API_KEY)}, "
            f"GOOGLE_API_KEY={_mask(self.GOOGLE_API_KEY)}, "
            f"CHROMA_PERSIST_DIR={self.CHROMA_PERSIST_DIR!r}, "
            f"DEFAULT_MODEL={self.DEFAULT_MODEL!r}"
            f")"
        )


# Singleton instance — import this everywhere:
#   from config import settings
settings = Settings()
