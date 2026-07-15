"""BM25 lexical retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from rank_bm25 import BM25Okapi

from ..chunking import TextChunk
from .base import validate_query, validate_top_k
from .corpus import Corpus
from .models import RetrievalResult, rank_results
from .tokenize import tokenize

__all__ = ["BM25Retriever"]

Tokenizer = Callable[[str], list[str]]


class BM25Retriever:
    """Ranks chunks by BM25 over their tokenized text.

    BM25 is the strong lexical baseline: it rewards rare query terms, saturates on
    repeated ones, and normalizes for chunk length. It is unbeatable on exact
    terminology, acronyms, and identifiers, and helpless against paraphrase — which is
    precisely the gap dense retrieval is added to cover.

    The index is built once at construction. The corpus is never re-tokenized per
    query; only the query is.

    Scoring itself comes from ``rank_bm25``, a direct implementation of the standard
    Okapi formula. The algorithms this project implements explicitly are the ones a
    reader is meant to verify by eye — rank fusion and the metrics — not a textbook
    weighting scheme with a well-tested library behind it.
    """

    def __init__(self, corpus: Corpus, *, tokenizer: Tokenizer = tokenize) -> None:
        self._corpus = corpus
        self._tokenizer = tokenizer

        tokenized = [tokenizer(chunk.text) for chunk in corpus]
        if not any(tokenized):
            raise ValueError(
                "every chunk tokenized to nothing; the corpus has no searchable text"
            )
        # rank_bm25 divides by the average document length, which is zero if every
        # document is empty — guarded above. Individual empty documents are fine.
        self._index = BM25Okapi(tokenized)

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        """Return the ``top_k`` best-scoring chunks for ``query``.

        Chunks scoring zero — no query term appears in them at all — are dropped rather
        than padded in. Returning a chunk with no lexical overlap would make Recall@K
        look better than the retriever earned.
        """
        validate_query(query)
        validate_top_k(top_k)

        scores = self._index.get_scores(self._tokenizer(query))
        scored: Sequence[tuple[TextChunk, float]] = [
            (chunk, float(score))
            for chunk, score in zip(self._corpus.chunks, scores, strict=True)
            if score > 0.0
        ]
        return rank_results(scored, top_k=top_k)
