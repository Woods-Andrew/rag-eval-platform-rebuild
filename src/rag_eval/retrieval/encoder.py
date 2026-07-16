"""The embedding-model boundary: a protocol, and the real adapter behind it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = ["SentenceTransformerEncoder", "TextEncoder", "l2_normalize"]

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class TextEncoder(Protocol):
    """Turns text into vectors.

    This exists so no test ever downloads a model. Every consumer takes an encoder
    rather than a model name, so a fake with fixed vectors substitutes cleanly and the
    unit suite stays offline and deterministic.
    """

    def encode(self, texts: Sequence[str]) -> Any:
        """Embed ``texts``, returning an ``(len(texts), dimension)`` array."""
        ...


def l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale each row to unit length, leaving all-zero rows alone.

    Once every vector is unit length, a dot product *is* the cosine similarity, so the
    whole search reduces to one matrix multiply. Zero rows — an empty chunk, or a query
    of entirely out-of-vocabulary tokens — would divide by zero, so they pass through
    and simply score zero against everything.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


class SentenceTransformerEncoder:
    """Adapter over a ``sentence-transformers`` bi-encoder.

    The library is imported lazily inside ``__init__`` rather than at module import.
    Importing torch costs seconds, and nothing in the offline test suite should pay
    that price — or be able to reach the network by accident.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed ``texts`` in batches, returning a float32 array."""
        embeddings = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def __repr__(self) -> str:
        return f"SentenceTransformerEncoder({self.model_name!r})"
