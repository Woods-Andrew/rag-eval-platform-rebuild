"""Dense embedding retrieval by cosine similarity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import validate_query, validate_top_k
from .cache import EmbeddingCache, encoder_identity, fingerprint
from .corpus import Corpus
from .encoder import TextEncoder, l2_normalize
from .models import RetrievalResult, rank_results

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = ["DenseRetriever"]


class DenseRetriever:
    """Ranks chunks by cosine similarity between their embedding and the query's.

    Where BM25 matches terms, this matches meaning: a query about "handling absent
    measurements" can rank a passage about "missing modality imputation" that shares no
    vocabulary with it. The tradeoff runs the other way too — embeddings blur precise
    tokens, so a rare identifier or an exact metric name can lose to a passage that
    merely sounds similar.

    The corpus is embedded **once**, at construction. Re-embedding per query would be
    the single most expensive mistake available here, and it is the reason the encoder
    is injected rather than constructed internally: the caller owns the model and can
    reuse it across retrievers. Passing an ``EmbeddingCache`` extends that across
    processes, so a second run of the same document and model reads the matrix from
    disk instead of recomputing it.

    Search is exact, not approximate. At this corpus size a full matrix multiply is
    both correct and fast, and an ANN index would obscure the mechanics this project
    exists to show.
    """

    def __init__(
        self, corpus: Corpus, encoder: TextEncoder, *, cache: EmbeddingCache | None = None
    ) -> None:
        self._corpus = corpus
        self._encoder = encoder
        self._cache_hit = False

        self._embeddings: NDArray[np.float32] = l2_normalize(self._corpus_embeddings(cache))

    def _corpus_embeddings(self, cache: EmbeddingCache | None) -> NDArray[np.float32]:
        """Load the corpus matrix from ``cache`` if it is there, otherwise encode it.

        A cache miss and no cache at all take the same path, so behaviour does not
        depend on whether caching is enabled — only how long it takes.
        """
        chunk_ids = self._corpus.chunk_ids
        key = fingerprint(encoder_identity(self._encoder), chunk_ids) if cache else ""

        if cache is not None:
            cached = cache.load(key, chunk_ids)
            if cached is not None:
                self._cache_hit = True
                return self._as_matrix(cached)

        embeddings = self._as_matrix(self._encoder.encode([chunk.text for chunk in self._corpus]))
        if embeddings.shape[0] != len(self._corpus):
            raise ValueError(
                f"encoder returned {embeddings.shape[0]} vectors for {len(self._corpus)} chunks"
            )

        if cache is not None:
            cache.store(key, chunk_ids, embeddings)
        return embeddings

    @property
    def loaded_from_cache(self) -> bool:
        """True when the corpus matrix came off disk rather than out of the encoder."""
        return self._cache_hit

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    @property
    def dimension(self) -> int:
        """Width of the embedding space this retriever was built in."""
        return int(self._embeddings.shape[1])

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        """Return the ``top_k`` chunks most similar to ``query``.

        Only the query is encoded here; the corpus embeddings were computed once at
        construction and are reused for every call.
        """
        validate_query(query)
        validate_top_k(top_k)

        query_vector = self._as_matrix(self._encoder.encode([query]))
        if query_vector.shape != (1, self.dimension):
            raise ValueError(
                f"encoder returned {query_vector.shape} for one query; "
                f"expected (1, {self.dimension})"
            )

        # Both sides are unit length, so this dot product is the cosine similarity.
        scores = self._embeddings @ l2_normalize(query_vector)[0]
        return rank_results(zip(self._corpus.chunks, scores, strict=True), top_k=top_k)

    @staticmethod
    def _as_matrix(encoded: object) -> NDArray[np.float32]:
        """Coerce whatever the encoder returned into a 2-D float32 array."""
        matrix = np.asarray(encoded, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"encoder must return a 2-D array, got shape {matrix.shape}")
        if matrix.shape[1] == 0:
            raise ValueError("encoder returned zero-width vectors")
        return matrix
