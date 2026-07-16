"""Deterministic stand-ins for the ML models, so the unit suite stays offline.

These are the reason encoders and rerankers are dependency-injected: nothing here
downloads a model, touches the network, or varies between runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = ["FakeEncoder", "FakeReranker"]


class FakeEncoder:
    """Maps exact strings to fixed vectors; anything unknown embeds to zeros.

    Zero vectors are deliberate rather than an accident of the fake: they score zero
    against everything, which is exactly how an out-of-vocabulary query should behave.
    """

    def __init__(self, vectors: Mapping[str, Sequence[float]], *, dimension: int) -> None:
        self.dimension = dimension
        self.vectors = {text: np.asarray(vector, dtype=np.float32) for text, vector in vectors.items()}
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        batch = list(texts)
        self.calls.append(batch)
        return np.asarray(
            [self.vectors.get(text, np.zeros(self.dimension, dtype=np.float32)) for text in batch],
            dtype=np.float32,
        )

    @property
    def encoded_texts(self) -> list[str]:
        """Every text passed to :meth:`encode`, flattened across calls."""
        return [text for batch in self.calls for text in batch]


class FakeReranker:
    """Scores a (query, passage) pair from a lookup table, defaulting to zero."""

    def __init__(self, scores: Mapping[tuple[str, str], float]) -> None:
        self.scores = dict(scores)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        self.calls.append((query, tuple(passages)))
        return [float(self.scores.get((query, passage), 0.0)) for passage in passages]
