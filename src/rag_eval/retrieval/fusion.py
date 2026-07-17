"""Reciprocal rank fusion, written out explicitly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import RetrievalResult

__all__ = ["DEFAULT_RRF_K", "reciprocal_rank_fusion", "rrf_scores"]

DEFAULT_RRF_K = 60


def rrf_scores(
    ranked_lists: Sequence[Sequence[RetrievalResult]], *, k: int = DEFAULT_RRF_K
) -> dict[str, float]:
    """Fused score per chunk ID:

    ::

        RRF(d) = sum over retrievers i of  1 / (k + rank_i(d))

    Only *rank position* is used. The retrievers' own scores are deliberately never
    read here: BM25 scores are unbounded and corpus-dependent, cosine similarities live
    in [-1, 1], and adding or min-max normalizing them would make the weighting an
    artifact of score distributions rather than of retrieval quality.

    A chunk missing from a retriever's list simply contributes nothing from that
    retriever — there is no imputed rank, and no penalty beyond the absent term.

    ``k`` damps the influence of the very top ranks. At k=60 the gap between rank 1 and
    rank 2 is small, so one retriever's favourite cannot unilaterally decide the fused
    order; a chunk both retrievers rank highly beats a chunk either one loves alone.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    fused: dict[str, float] = {}
    for results in ranked_lists:
        for result in results:
            fused[result.chunk_id] = fused.get(result.chunk_id, 0.0) + 1.0 / (k + result.rank)
    return fused


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    *,
    top_k: int,
    k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    """Fuse ranked lists into one, best first, renumbered from rank 1.

    Ties break on chunk ID so the fused order does not depend on which retriever was
    passed first.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    scores = rrf_scores(ranked_lists, k=k)
    chunks: Mapping[str, RetrievalResult] = {
        result.chunk_id: result for results in ranked_lists for result in results
    }

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        RetrievalResult(chunk=chunks[chunk_id].chunk, score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(ordered[:top_k], start=1)
    ]
