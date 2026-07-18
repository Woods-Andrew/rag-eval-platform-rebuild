"""Recall@K and nDCG@K, written out explicitly."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

__all__ = ["dcg", "ndcg_at_k", "recall_at_k"]


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top ``k`` retrieved.

    ::

        Recall@K = |retrieved[:K] ∩ relevant| / |relevant|

    This is the ceiling question: if a passage is not in the retrieved set, no amount
    of prompt engineering recovers it. Position within the top K does not matter here —
    that is what nDCG is for.

    The denominator is the number of *known relevant* chunks, so a query with three
    labels can only reach 1.0 by finding all three. A query with no labels raises: it
    cannot be scored, and silently returning 0.0 or 1.0 would quietly bias the mean.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        raise ValueError("cannot compute recall for a query with no relevant chunks")

    relevant_set = set(relevant)
    found = sum(1 for chunk_id in retrieved[:k] if chunk_id in relevant_set)
    return found / len(relevant_set)


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of a gain sequence, 1-indexed.

    ::

        DCG = Σ  gain_i / log2(i + 1)
             i=1..n

    The rank-1 discount is ``log2(2) == 1``, so the top hit is undiscounted.
    """
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Normalized discounted cumulative gain over the top ``k`` retrieved.

    ::

        nDCG@K = DCG@K / IDCG@K

    Where Recall@K asks *did we find it*, this asks *did we rank it well*: each hit is
    discounted logarithmically by its position, then normalized against the best
    possible ordering so the result lands in [0, 1].

    Relevance is binary — gain 1 for a labelled chunk, 0 otherwise — which is the
    honest choice for a hand-labelled benchmark of this size. The ideal ranking is
    therefore every relevant chunk packed into the top positions, capped at ``k``
    because no ranking of length ``k`` can surface more than ``k`` of them.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not relevant:
        raise ValueError("cannot compute nDCG for a query with no relevant chunks")

    relevant_set = set(relevant)
    gains = [1.0 if chunk_id in relevant_set else 0.0 for chunk_id in retrieved[:k]]

    ideal_hits = min(len(relevant_set), k)
    ideal_dcg = dcg([1.0] * ideal_hits)
    if ideal_dcg == 0.0:  # pragma: no cover - unreachable while relevant is non-empty
        return 0.0
    return dcg(gains) / ideal_dcg
