"""Chunking: pages become the units the retriever actually ranks."""

from __future__ import annotations

from .base import Chunker
from .fixed import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    FixedSizeChunker,
    split_into_word_windows,
    word_count,
)
from .models import TextChunk, chunk_id_for
from .sentences import split_sentences
from .structure import (
    DEFAULT_MAX_WORDS,
    DEFAULT_MIN_WORDS,
    StructureAwareChunker,
    is_heading,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_MAX_WORDS",
    "DEFAULT_MIN_WORDS",
    "DEFAULT_OVERLAP",
    "Chunker",
    "FixedSizeChunker",
    "StructureAwareChunker",
    "TextChunk",
    "chunk_id_for",
    "is_heading",
    "split_into_word_windows",
    "split_sentences",
    "word_count",
]
