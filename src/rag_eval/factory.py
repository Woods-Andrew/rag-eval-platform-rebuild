"""Building the four retrieval strategies over one corpus, each at most once.

The CLI and the UI both need "give me the reranked retriever for this corpus, and do
not rebuild anything you already built". That logic is identical for both and easy to
get subtly wrong in two places, so it lives here.
"""

from __future__ import annotations

from collections.abc import Callable

from .retrieval import (
    BM25Retriever,
    Corpus,
    DenseRetriever,
    EmbeddingCache,
    HybridRetriever,
    Retriever,
    TextEncoder,
)

__all__ = ["RETRIEVER_NAMES", "RetrieverFactory"]

RETRIEVER_NAMES = ("bm25", "dense", "hybrid", "reranked")

EncoderFactory = Callable[[], TextEncoder]
RerankerFactory = Callable[[], object]


class RetrieverFactory:
    """Constructs retrievers over a fixed corpus, memoizing everything it builds.

    Two costs are avoided, and they are different costs. Memoizing the *encoder* avoids
    loading the model weights repeatedly; memoizing the *dense retriever* avoids
    embedding the corpus repeatedly. Evaluating all four strategies without this pays
    both four times over, and hybrid and reranked are built from the same components as
    dense — so the strategies genuinely share work rather than merely resembling each
    other.

    The models arrive as factories so nothing is loaded until a strategy that needs one
    is asked for, and so tests can substitute fakes.
    """

    def __init__(
        self,
        corpus: Corpus,
        *,
        cache: EmbeddingCache | None = None,
        encoder_factory: EncoderFactory | None = None,
        reranker_factory: RerankerFactory | None = None,
    ) -> None:
        self._corpus = corpus
        self._cache = cache
        self._encoder_factory = encoder_factory or _default_encoder
        self._reranker_factory = reranker_factory or _default_reranker
        self._retrievers: dict[str, Retriever] = {}
        self._shared_encoder: TextEncoder | None = None

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    @property
    def built(self) -> tuple[str, ...]:
        """Which strategies have actually been constructed, in construction order."""
        return tuple(self._retrievers)

    def get(self, name: str) -> Retriever:
        """Return the named retriever, building it once and reusing it thereafter."""
        if name not in RETRIEVER_NAMES:
            raise ValueError(
                f"unknown retriever {name!r}; expected one of {', '.join(RETRIEVER_NAMES)}"
            )

        if name not in self._retrievers:
            self._retrievers[name] = self._build(name)
        return self._retrievers[name]

    def all(self, names: tuple[str, ...] | None = None) -> dict[str, Retriever]:
        """Build several strategies, sharing every component they have in common."""
        return {name: self.get(name) for name in (names or RETRIEVER_NAMES)}

    def _build(self, name: str) -> Retriever:
        if name == "bm25":
            return BM25Retriever(self._corpus)
        if name == "dense":
            return DenseRetriever(self._corpus, self._encoder(), cache=self._cache)
        if name == "hybrid":
            # Built through ``get`` so selecting hybrid after dense reuses the
            # embeddings rather than computing them a second time.
            return HybridRetriever([self.get("bm25"), self.get("dense")])

        from .reranking import RerankingRetriever

        return RerankingRetriever(
            self.get("hybrid"),
            self._reranker_factory(),  # type: ignore[arg-type]
        )

    def _encoder(self) -> TextEncoder:
        """One encoder for the whole factory; a second copy would double the memory."""
        if self._shared_encoder is None:
            self._shared_encoder = self._encoder_factory()
        return self._shared_encoder


def _default_encoder() -> TextEncoder:
    from .retrieval import SentenceTransformerEncoder

    return SentenceTransformerEncoder()


def _default_reranker() -> object:
    from .reranking import CrossEncoderReranker

    return CrossEncoderReranker()
