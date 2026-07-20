"""The cross-encoder boundary: a protocol, and the real adapter behind it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["CrossEncoderReranker", "DEFAULT_RERANK_MODEL", "Reranker"]

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    """Scores how well each passage answers a query.

    Injected for the same reason encoders are: no unit test downloads a model. Scores
    are only meaningful relative to each other within a single call — a cross-encoder
    emits an unbounded logit, not a probability.
    """

    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]:
        """Return one score per passage, in the order given."""
        ...


class CrossEncoderReranker:
    """Adapter over a ``sentence-transformers`` cross-encoder.

    A bi-encoder embeds query and passage separately, so it can never model term-level
    interaction between them. A cross-encoder reads the pair jointly and is
    substantially more accurate — and far too slow to run over a whole corpus, which is
    exactly why it sits behind a cheap retriever rather than replacing one.

    ``sentence_transformers`` is imported lazily inside ``__init__`` so importing this
    package costs nothing and cannot reach the network by accident.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.model_name = model_name
        self._batch_size = batch_size
        self._model = CrossEncoder(model_name, device=device)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Score every ``(query, passage)`` pair jointly."""
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def __repr__(self) -> str:
        return f"CrossEncoderReranker({self.model_name!r})"
