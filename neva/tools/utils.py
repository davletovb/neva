"""Shared utilities for tool implementations."""

from __future__ import annotations


def missing_dependency_message(package: str, *, fallback: str) -> str:
    """Return a helpful error message for optional tool dependencies."""

    return (
        f"The optional dependency `{package}` is unavailable. {fallback} "
        f"Install it with `pip install {package}` (or `pip install neva[tools]`) "
        "or provide a custom factory when constructing the tool to continue "
        "running experiments without network access."
    )
