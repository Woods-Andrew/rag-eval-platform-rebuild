"""A retriever that reorders another retriever's candidates with a cross-encoder."""

from __future__ import annotations

from ..retrieval import RetrievalResult, Retriever, validate_query, validate_top_k
from .base import Reranker

__all__ = ["DEFAULT_CANDIDATE_MULTIPLIER", "RerankingRetriever"]

# The reranker can only reorder what it is handed, so it is given several times the
# requested depth to work with.
DEFAULT_CANDIDATE_MULTIPLIER = 5


class RerankingRetriever:
    """Retrieve cheaply and deeply, then rescore the candidates jointly.

    This is the standard two-stage shape, and the reason it exists is cost: scoring
    every (query, chunk) pair with a cross-encoder is quadratic in corpus size and
    hopeless at scale, while scoring fifty candidates is trivial.

    It follows that **reranking can never improve Recall@K beyond what the base
    retriever already surfaced within the candidate window** — a chunk the first stage
    never returned cannot be promoted. What it improves is ordering, which is what nDCG
    measures. A reranker that lifts nDCG while leaving recall flat is working exactly
    as intended, not underperforming.

    Being a ``Retriever`` itself, it composes: wrap a hybrid, and the evaluator measures
    it through the same path as everything else.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError(
                f"candidate_multiplier must be at least 1, got {candidate_multiplier}"
            )

        self._retriever = retriever
        self._reranker = reranker
        self._candidate_multiplier = candidate_multiplier

    @property
    def base_retriever(self) -> Retriever:
        return self._retriever

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        """Fetch ``top_k * candidate_multiplier`` candidates, then reorder them."""
        query = validate_query(query)
        validate_top_k(top_k)

        candidates = self._retriever.retrieve(
            query, top_k=top_k * self._candidate_multiplier
        )
        if not candidates:
            return []

        scores = self._reranker.score(query, [result.chunk.text for result in candidates])
        if len(scores) != len(candidates):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(candidates)} passages"
            )

        # Ties break on chunk_id so the output does not depend on the candidate order
        # the first stage happened to produce.
        ordered = sorted(
            zip(candidates, scores, strict=True),
            key=lambda pair: (-pair[1], pair[0].chunk_id),
        )
        return [
            RetrievalResult(chunk=result.chunk, score=float(score), rank=rank)
            for rank, (result, score) in enumerate(ordered[:top_k], start=1)
        ]
