"""Hybrid retrieval: several retrievers, fused over ranks."""

from __future__ import annotations

from collections.abc import Sequence

from .base import Retriever, validate_query, validate_top_k
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from .models import RetrievalResult

__all__ = ["DEFAULT_CANDIDATE_MULTIPLIER", "HybridRetriever"]

# Each retriever is asked for more candidates than the caller wants, so a chunk one
# retriever ranks 20th still has a rank the other's vote can lift into the top 5.
DEFAULT_CANDIDATE_MULTIPLIER = 4


class HybridRetriever:
    """Combines any number of retrievers by reciprocal rank fusion.

    It knows nothing about its members beyond the ``Retriever`` protocol, so lexical,
    dense, and any future strategy compose without special-casing — and the same class
    is what the benchmark measures against its own parts.

    Each member is queried for ``top_k * candidate_multiplier`` results before fusion.
    Fusing only the top ``top_k`` from each would throw away exactly the evidence
    fusion exists to use: a chunk ranked 12th by BM25 and 3rd by dense is precisely the
    case where the two disagree and the fused answer beats both.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        k: int = DEFAULT_RRF_K,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        if not retrievers:
            raise ValueError("a hybrid retriever needs at least one member retriever")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if candidate_multiplier < 1:
            raise ValueError(
                f"candidate_multiplier must be at least 1, got {candidate_multiplier}"
            )

        self._retrievers = tuple(retrievers)
        self._k = k
        self._candidate_multiplier = candidate_multiplier

    @property
    def retrievers(self) -> tuple[Retriever, ...]:
        return self._retrievers

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        """Query every member, then fuse their rankings into one."""
        validate_query(query)
        validate_top_k(top_k)

        candidates = top_k * self._candidate_multiplier
        ranked_lists = [
            retriever.retrieve(query, top_k=candidates) for retriever in self._retrievers
        ]
        return reciprocal_rank_fusion(ranked_lists, top_k=top_k, k=self._k)
