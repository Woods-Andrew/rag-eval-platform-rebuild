"""What a retriever returns."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..chunking import TextChunk

__all__ = ["RetrievalResult", "rank_results"]


@dataclass(frozen=True)
class RetrievalResult:
    """One chunk a retriever returned, with its score and its position in the ranking.

    ``rank`` is 1-indexed because that is what the reciprocal rank fusion and nDCG
    formulas expect; an off-by-one here silently changes every published metric.

    ``score`` is only comparable *within* one retriever's output. BM25 scores are
    unbounded and corpus-dependent, cosine similarities live in [-1, 1], and fused
    scores are neither — which is exactly why fusion happens over ranks.
    """

    chunk: TextChunk
    score: float
    rank: int

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank is 1-indexed, got {self.rank}")

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def citation(self) -> str:
        """Human-readable provenance for this hit, e.g. ``omics.pdf p.3 § Methods``."""
        return self.chunk.citation


def rank_results(
    scored: Iterable[tuple[TextChunk, float]], *, top_k: int
) -> list[RetrievalResult]:
    """Order scored chunks best-first and keep the top ``top_k``.

    Ties break on ``chunk_id``, not on corpus order. Two chunks with identical scores
    would otherwise swap places depending on how the corpus happened to be built, and a
    benchmark whose numbers move when the input order changes is not measuring
    retrieval quality.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    ordered = sorted(scored, key=lambda pair: (-pair[1], pair[0].chunk_id))
    return [
        RetrievalResult(chunk=chunk, score=float(score), rank=rank)
        for rank, (chunk, score) in enumerate(ordered[:top_k], start=1)
    ]
