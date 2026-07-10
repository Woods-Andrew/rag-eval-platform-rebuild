"""Fixed-size chunking: pages become overlapping windows of a fixed word count."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..ingestion import PDFPage
from .models import TextChunk, chunk_id_for

__all__ = ["FixedSizeChunker"]

DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 40

_WORD = re.compile(r"\S+")


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
        spans = [match.span() for match in _WORD.finditer(page.text)]
        if not spans:
            return []

        return [
            self._build_chunk(page, index, spans[start:end])
            for index, (start, end) in enumerate(self._windows(len(spans)))
        ]

    def _windows(self, word_count: int) -> list[tuple[int, int]]:
        """Half-open ``[start, end)`` word ranges covering ``word_count`` words.

        The loop stops as soon as a window reaches the end of the page, so the tail is
        never emitted twice as a shorter chunk fully contained in its predecessor.
        """
        windows: list[tuple[int, int]] = []
        start = 0
        while start < word_count:
            end = min(start + self.chunk_size, word_count)
            windows.append((start, end))
            if end == word_count:
                break
            start += self.stride
        return windows

    def _build_chunk(
        self, page: PDFPage, index: int, spans: Sequence[tuple[int, int]]
    ) -> TextChunk:
        """Slice the page's original text so intra-chunk formatting survives verbatim."""
        text = page.text[spans[0][0] : spans[-1][1]]
        return TextChunk(
            chunk_id=chunk_id_for(page.source, page.page_number, index, text),
            text=text,
            source=page.source,
            page_number=page.page_number,
            chunk_index=index,
            metadata=page.metadata,
        )
