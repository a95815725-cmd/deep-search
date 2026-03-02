"""
Evaluation metrics for the Financial Deep Search agent.

  - detect_hallucinations      : numbers in answer not found in source docs
  - calculate_answer_completeness : fraction of expected key facts present
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RE_NUMBERS = re.compile(
    r"(?<!\w)(\(?)(-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?)(\)?)(?!\w)",
    re.VERBOSE,
)


def _extract_numbers(text: str) -> Set[str]:
    nums: Set[str] = set()
    for m in _RE_NUMBERS.finditer(text):
        raw = m.group(2).replace(",", "").replace(" ", "").strip()
        if raw and raw not in {"0", "1"}:
            nums.add(raw)
    return nums


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# 1. Hallucination detection
# ---------------------------------------------------------------------------


def detect_hallucinations(
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Flag numbers in the answer that don't appear in any source document.

    Returns dict with:
      hallucinated_numbers, verified_numbers, hallucination_rate,
      total_answer_numbers, source_number_count
    """
    if not answer:
        return {
            "hallucinated_numbers": [],
            "verified_numbers": [],
            "hallucination_rate": 0.0,
            "total_answer_numbers": 0,
            "source_number_count": 0,
        }

    source_numbers: Set[str] = set()
    for doc in retrieved_docs:
        text = doc.get("text") or doc.get("content") or doc.get("page_content") or ""
        source_numbers.update(_extract_numbers(text))

    answer_numbers = _extract_numbers(answer)
    hallucinated: List[str] = []
    verified: List[str] = []

    for num in sorted(answer_numbers):
        if num in source_numbers:
            verified.append(num)
        else:
            # Allow 1% rounding tolerance
            try:
                float_val = float(num)
                close = any(
                    abs(float(s) - float_val) / max(abs(float_val), 1e-9) < 0.01
                    for s in source_numbers
                    if _is_numeric(s)
                )
                if close:
                    verified.append(num)
                    continue
            except ValueError:
                pass
            hallucinated.append(num)

    total = len(answer_numbers)
    return {
        "hallucinated_numbers": hallucinated,
        "verified_numbers": verified,
        "hallucination_rate": round(len(hallucinated) / total, 4) if total else 0.0,
        "total_answer_numbers": total,
        "source_number_count": len(source_numbers),
    }


# ---------------------------------------------------------------------------
# 2. Answer completeness
# ---------------------------------------------------------------------------


def calculate_answer_completeness(
    answer: str,
    ground_truth_hints: List[str],
) -> float:
    """
    Fraction of expected key facts/phrases present in the answer (case-insensitive).
    Returns 1.0 if hints list is empty.
    """
    if not ground_truth_hints:
        return 1.0
    if not answer:
        return 0.0

    norm_answer = _normalise(answer)
    found = sum(1 for hint in ground_truth_hints if _normalise(hint) in norm_answer)
    return round(found / len(ground_truth_hints), 4)
