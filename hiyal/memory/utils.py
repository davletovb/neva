"""Utility helpers shared across memory implementations."""

from __future__ import annotations

from math import sqrt
from typing import Sequence


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute the cosine similarity between two vectors."""

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sqrt(sum(a * a for a in vec_a))
    mag_b = sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def estimate_token_count(text: str) -> int:
    """Rudimentary token estimator used for budgeting heuristics."""

    if not text:
        return 0
    return max(1, len(text.split()))


__all__ = ["cosine_similarity", "estimate_token_count"]
