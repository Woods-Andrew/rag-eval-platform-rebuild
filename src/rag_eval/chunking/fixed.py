"""Fixed-size chunking: pages become overlapping windows of a fixed word count."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..ingestion import PDFPage
from .models import TextChunk, chunk_id_for

__all__ = ["FixedSizeChunker", "split_into_word_windows", "word_count"]

DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 40

_WORD = re.compile(r"\S+")


def word_count(text: str) -> int:
    """Words in ``text``, the unit every chunk size in this package is expressed in."""
    return len(text.split())


def split_into_word_windows(text: str, *, size: int, overlap: int = 0) -> list[str]:
    """Slice ``text`` into windows of at most ``size`` words, overlapping by ``overlap``.

    Windows are sliced out of ``text`` rather than rejoined from split words, so
    whitespace and line structure inside a window survive verbatim. The final window is
    whatever is left over; it is never emitted as a duplicate of its predecessor.

    Returns an empty list for text with no words.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got {overlap} of {size}")

    spans = [match.span() for match in _WORD.finditer(text)]
    if not spans:
        return []

    stride = size - overlap
    windows: list[str] = []
    start = 0
    while start < len(spans):
        end = min(start + size, len(spans))
        windows.append(text[spans[start][0] : spans[end - 1][1]])
        if end == len(spans):
            break
        start += stride
    return windows


@dataclass(frozen=True)
class FixedSizeChunker:
    """Split pages into overlapping windows of ``chunk_size`` words.

    Sizes are counted in **words**, not characters: word counts track token counts far
    more closely than character counts do, BM25 tokenizes on word boundaries anyway, and
    a window that ends on a word boundary can never cut an acronym or identifier in half.

    Windows never cross a page boundary, so every chunk cites exactly one page. The cost
    is that the final window on a page is usually short, and a sentence straddling a page
    break is split — accepted, because an unambiguous citation is worth more here than a
    tidy sentence.

    ``overlap`` words are repeated between consecutive windows so that a passage landing
    on a window boundary still appears intact in one of them.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    overlap: int = DEFAULT_OVERLAP

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must not be negative, got {self.overlap}")
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be smaller than chunk_size "
                f"({self.chunk_size}); otherwise the window never advances"
            )

    @property
    def stride(self) -> int:
        """Words the window advances between chunks."""
        return self.chunk_size - self.overlap

    def chunk_pages(self, pages: Iterable[PDFPage]) -> list[TextChunk]:
        """Chunk a whole document, in page order.

        Pages with no extractable text produce no chunks; page numbering is untouched,
        so a gap in the chunked output is a real blank page rather than a lost one.
        """
        return [chunk for page in pages for chunk in self.chunk_page(page)]

    def chunk_page(self, page: PDFPage) -> list[TextChunk]:
        """Chunk a single page. Returns an empty list for a page with no text."""
        windows = split_into_word_windows(
            page.text, size=self.chunk_size, overlap=self.overlap
        )
        return [
            TextChunk(
                chunk_id=chunk_id_for(page.source, page.page_number, index, text),
                text=text,
                source=page.source,
                page_number=page.page_number,
                chunk_index=index,
                metadata=page.metadata,
            )
            for index, text in enumerate(windows)
        ]
