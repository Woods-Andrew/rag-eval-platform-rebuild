"""Data model for a retrievable chunk of text."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

__all__ = ["TextChunk", "chunk_id_for"]


def chunk_id_for(source: str, page_number: int, chunk_index: int, text: str) -> str:
    """Build the stable identifier for a chunk: readable provenance plus a content digest.

    The readable prefix (``paper-p003-c02``) lets a human scanning a relevance-label file
    see which page a label points at without opening the corpus. The trailing digest makes
    the ID a function of the chunk's *text*, so re-chunking with different settings — or
    re-ingesting an edited PDF — produces different IDs. A stale relevance label then fails
    to match anything instead of silently pointing at text it was never written against,
    which is the failure mode that would quietly corrupt a benchmark.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{Path(source).stem}-p{page_number:03d}-c{chunk_index:02d}-{digest}"


@dataclass(frozen=True)
class TextChunk:
    """A unit of text as the retriever sees it, carrying its provenance forward.

    A chunk never spans pages, so ``page_number`` is a single unambiguous citation
    target. ``chunk_index`` is the chunk's 0-based ordinal *within its page*, which
    together with ``page_number`` gives a total ordering over the document.
    """

    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must be a non-empty filename")
        if not self.text.strip():
            raise ValueError("a chunk must contain text; empty chunks are dropped, not stored")
        if self.page_number < 1:
            raise ValueError(f"page_number is 1-indexed, got {self.page_number}")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index is 0-based, got {self.chunk_index}")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def word_count(self) -> int:
        """Number of whitespace-separated words, the unit chunk sizes are expressed in."""
        return len(self.text.split())

    @property
    def citation(self) -> str:
        """Short human-readable provenance, e.g. ``paper.pdf p.3``."""
        return f"{self.source} p.{self.page_number}"
