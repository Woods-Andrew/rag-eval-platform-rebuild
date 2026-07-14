"""The searchable collection of chunks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

from ..chunking import TextChunk

__all__ = ["Corpus"]


class Corpus:
    """An ordered collection of chunks with unique IDs, addressable by ID.

    The uniqueness check is a benchmark-integrity guard, not a formality: relevance
    labels reference chunks by ID, so a duplicate ID means a label points at two
    different passages and the resulting Recall@K is meaningless. Better to fail at
    construction than to publish a number built on it.
    """

    def __init__(self, chunks: Iterable[TextChunk]) -> None:
        self._chunks: tuple[TextChunk, ...] = tuple(chunks)
        if not self._chunks:
            raise ValueError("a corpus needs at least one chunk")

        self._by_id: dict[str, TextChunk] = {}
        for chunk in self._chunks:
            if chunk.chunk_id in self._by_id:
                raise ValueError(f"duplicate chunk_id in corpus: {chunk.chunk_id}")
            self._by_id[chunk.chunk_id] = chunk

    @property
    def chunks(self) -> Sequence[TextChunk]:
        """The chunks, in the order they were supplied."""
        return self._chunks

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(chunk.chunk_id for chunk in self._chunks)

    @property
    def sources(self) -> tuple[str, ...]:
        """Distinct source documents, in first-seen order."""
        return tuple(dict.fromkeys(chunk.source for chunk in self._chunks))

    def get(self, chunk_id: str) -> TextChunk:
        """Look up a chunk by ID, raising ``KeyError`` when it is not in the corpus."""
        try:
            return self._by_id[chunk_id]
        except KeyError:
            raise KeyError(f"no chunk with id {chunk_id!r} in this corpus") from None

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._by_id

    def __len__(self) -> int:
        return len(self._chunks)

    def __iter__(self) -> Iterator[TextChunk]:
        return iter(self._chunks)

    def __repr__(self) -> str:
        return f"Corpus({len(self)} chunks from {len(self.sources)} source(s))"
