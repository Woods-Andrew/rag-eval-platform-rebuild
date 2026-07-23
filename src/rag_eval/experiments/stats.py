"""Describing a corpus numerically, independently of any relevance labels."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..retrieval import Corpus

__all__ = ["CorpusStats", "describe_corpus"]


@dataclass(frozen=True)
class CorpusStats:
    """Shape of a chunked corpus: how many chunks, and how big they are.

    This is the half of a chunking experiment that needs no benchmark. Chunk-size
    distribution is a real, reportable property of a strategy — a chunker whose chunks
    range from 3 to 400 words is making a claim about retrieval that Recall@K will
    later either support or refute, and the distribution is measurable the moment a
    document exists.
    """

    chunk_count: int
    page_count: int
    total_words: int
    min_words: int
    median_words: float
    mean_words: float
    max_words: int

    @property
    def words_per_chunk(self) -> float:
        """Alias for ``mean_words``, spelled the way tables want to label it."""
        return self.mean_words

    def to_dict(self) -> dict[str, float | int]:
        return {
            "chunk_count": self.chunk_count,
            "page_count": self.page_count,
            "total_words": self.total_words,
            "min_words": self.min_words,
            "median_words": self.median_words,
            "mean_words": self.mean_words,
            "max_words": self.max_words,
        }


def describe_corpus(corpus: Corpus) -> CorpusStats:
    """Measure chunk-count and chunk-size distribution for ``corpus``.

    Sizes are word counts, matching the units the chunkers are configured in, so a
    ``max_words`` above a chunker's configured limit is immediately visible as the
    boundary case it is rather than hidden behind a character count.
    """
    counts = [len(chunk.text.split()) for chunk in corpus]
    pages = {(chunk.source, chunk.page_number) for chunk in corpus}

    return CorpusStats(
        chunk_count=len(counts),
        page_count=len(pages),
        total_words=sum(counts),
        min_words=min(counts),
        median_words=float(statistics.median(counts)),
        mean_words=sum(counts) / len(counts),
        max_words=max(counts),
    )
