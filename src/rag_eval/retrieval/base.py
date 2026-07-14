"""The interface every retrieval strategy satisfies, and the checks they all share."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import RetrievalResult

__all__ = ["Retriever", "validate_query", "validate_top_k"]


@runtime_checkable
class Retriever(Protocol):
    """Ranks a corpus against a query.

    Every retriever here — lexical, dense, fused, reranked — is interchangeable behind
    this one method, which is what lets the evaluator measure them all with the same
    code path and lets the hybrid retriever compose them without knowing what they are.
    """

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        """Return at most ``top_k`` results, best first, ranks starting at 1."""
        ...


def validate_query(query: str) -> str:
    """Reject an empty or whitespace-only query and return it stripped."""
    stripped = query.strip()
    if not stripped:
        raise ValueError("query must not be empty")
    return stripped


def validate_top_k(top_k: int) -> int:
    """Reject a non-positive ``top_k``."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    return top_k
