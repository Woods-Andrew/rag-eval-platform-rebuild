"""Structure-aware chunking: chunks follow the document's own section boundaries."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..ingestion import PDFPage
from .fixed import split_into_word_windows, word_count
from .models import TextChunk, chunk_id_for
from .sentences import split_sentences

__all__ = ["StructureAwareChunker", "is_heading"]

DEFAULT_MAX_WORDS = 250
DEFAULT_MIN_WORDS = 40

# A heading is short by nature; anything longer is running prose that merely looks
# like one.
_MAX_HEADING_WORDS = 12

# "3", "3.2", "3.2.1", optionally with a trailing period, then the heading text.
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")

_KNOWN_SECTIONS = frozenset(
    {
        "abstract",
        "acknowledgements",
        "acknowledgments",
        "appendix",
        "background",
        "conclusion",
        "conclusions",
        "discussion",
        "evaluation",
        "experimental setup",
        "experiments",
        "future work",
        "introduction",
        "limitations",
        "materials and methods",
        "method",
        "methodology",
        "methods",
        "references",
        "related work",
        "results",
        "results and discussion",
    }
)


def is_heading(line: str) -> bool:
    """True when a single line looks like a section heading rather than prose.

    Three signals, deliberately conservative: a known academic section name, a
    numbered heading (``3.2 Imputation Strategy``), or an all-caps line. Title-case
    detection is *not* used — it fires on figure captions and author lines often
    enough to be worse than missing a heading, and a missed heading degrades to
    paragraph-boundary chunking rather than producing a wrong chunk.
    """
    stripped = line.strip()
    if not stripped or word_count(stripped) > _MAX_HEADING_WORDS:
        return False
    if stripped.endswith((".", ",", ";")):
        return False

    if stripped.rstrip(":").lower() in _KNOWN_SECTIONS:
        return True
    if _NUMBERED_HEADING.match(stripped):
        return True

    letters = [character for character in stripped if character.isalpha()]
    return bool(letters) and all(character.isupper() for character in letters)


@dataclass(frozen=True)
class _Block:
    """A heading line, or one paragraph of body text."""

    text: str
    is_heading: bool


@dataclass(frozen=True)
class StructureAwareChunker:
    """Chunk pages along headings, paragraphs, and sentences, in that order.

    Fixed-size chunking cuts wherever the word count runs out, which routinely splits
    a definition from the term it defines. This strategy prefers the boundaries the
    author already put in the document, and only falls back to a cruder split when the
    text leaves it no choice:

    1. a heading always starts a new chunk, and tags every chunk beneath it;
    2. paragraphs accumulate until the next one would exceed ``max_words``;
    3. a single paragraph over ``max_words`` is split between sentences;
    4. a single sentence over ``max_words`` is split into fixed word windows.

    A trailing chunk shorter than ``min_words`` is merged back into its predecessor —
    a 12-word orphan retrieves poorly and pollutes the ranking. That merge is the one
    case where a chunk may exceed ``max_words``, by at most ``min_words - 1``.

    ``include_heading`` repeats the heading text at the top of each chunk beneath it.
    This costs a little duplication and buys a real retrieval signal: a chunk whose
    body never repeats the section topic still matches a query about it.
    """

    max_words: int = DEFAULT_MAX_WORDS
    min_words: int = DEFAULT_MIN_WORDS
    include_heading: bool = True

    def __post_init__(self) -> None:
        if self.max_words <= 0:
            raise ValueError(f"max_words must be positive, got {self.max_words}")
        if self.min_words < 0:
            raise ValueError(f"min_words must not be negative, got {self.min_words}")
        if self.min_words >= self.max_words:
            raise ValueError(
                f"min_words ({self.min_words}) must be smaller than max_words "
                f"({self.max_words})"
            )

    def chunk_pages(self, pages: Iterable[PDFPage]) -> list[TextChunk]:
        """Chunk a whole document, in page order.

        Sections are not carried across pages: a page that opens mid-section has no
        heading to attribute its first chunks to, and inventing one would be a guess.
        """
        return [chunk for page in pages for chunk in self.chunk_page(page)]

    def chunk_page(self, page: PDFPage) -> list[TextChunk]:
        """Chunk a single page. Returns an empty list for a page with no text."""
        segments = self._segment(page.text)
        return [
            TextChunk(
                chunk_id=chunk_id_for(page.source, page.page_number, index, text),
                text=text,
                source=page.source,
                page_number=page.page_number,
                chunk_index=index,
                section=section,
                metadata=page.metadata,
            )
            for index, (section, text) in enumerate(segments)
        ]

    def _segment(self, text: str) -> list[tuple[str | None, str]]:
        """Page text into ``(section, chunk text)`` pairs, in reading order."""
        chunks: list[tuple[str | None, list[str]]] = []
        section: str | None = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            if buffer:
                chunks.append((section, buffer))
                buffer = []

        for block in _blocks(text):
            if block.is_heading:
                flush()
                section = block.text
                continue
            for piece in self._fit(block.text):
                if buffer and word_count(" ".join(buffer)) + word_count(piece) > self.max_words:
                    flush()
                buffer.append(piece)
        flush()

        merged = self._merge_short_tails(chunks)
        return [(section, self._render(section, parts)) for section, parts in merged]

    def _merge_short_tails(
        self, chunks: list[tuple[str | None, list[str]]]
    ) -> list[tuple[str | None, list[str]]]:
        """Fold a runt final chunk into the previous chunk of the same section."""
        merged: list[tuple[str | None, list[str]]] = []
        for section, parts in chunks:
            too_short = word_count(" ".join(parts)) < self.min_words
            if merged and too_short and merged[-1][0] == section:
                merged[-1][1].extend(parts)
            else:
                merged.append((section, list(parts)))
        return merged

    def _render(self, section: str | None, parts: list[str]) -> str:
        """Join a chunk's paragraphs, optionally prefixing the section heading."""
        body = "\n\n".join(parts)
        if self.include_heading and section and not body.startswith(section):
            return f"{section}\n\n{body}"
        return body

    def _fit(self, paragraph: str) -> list[str]:
        """Break a paragraph down until every piece fits within ``max_words``."""
        if word_count(paragraph) <= self.max_words:
            return [paragraph]

        pieces: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer
            if buffer:
                pieces.append(" ".join(buffer))
                buffer = []

        for sentence in split_sentences(paragraph):
            if word_count(sentence) > self.max_words:
                flush()
                pieces.extend(split_into_word_windows(sentence, size=self.max_words))
                continue
            if buffer and word_count(" ".join(buffer)) + word_count(sentence) > self.max_words:
                flush()
            buffer.append(sentence)
        flush()
        return pieces


def _blocks(text: str) -> list[_Block]:
    """Split page text into heading lines and blank-line-separated paragraphs."""
    blocks: list[_Block] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(_Block("\n".join(paragraph), is_heading=False))
            paragraph.clear()

    for line in text.split("\n"):
        if not line.strip():
            flush()
        elif is_heading(line):
            flush()
            blocks.append(_Block(line.strip(), is_heading=True))
        else:
            paragraph.append(line)
    flush()
    return blocks
