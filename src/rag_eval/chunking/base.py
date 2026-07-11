"""The interface every chunking strategy satisfies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from ..ingestion import PDFPage
from .models import TextChunk

__all__ = ["Chunker"]


@runtime_checkable
class Chunker(Protocol):
    """Turns pages into chunks.

    This earns a Protocol because the benchmark compares chunking strategies against
    each other: the evaluation runner takes a ``Chunker``, not a specific class, so a
    new strategy is measurable without touching the runner.
    """

    def chunk_page(self, page: PDFPage) -> list[TextChunk]:
        """Chunk a single page. Empty pages yield no chunks."""
        ...

    def chunk_pages(self, pages: Iterable[PDFPage]) -> list[TextChunk]:
        """Chunk a document in page order."""
        ...
