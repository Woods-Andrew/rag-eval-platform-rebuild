"""Chunking: pages become the units the retriever actually ranks."""

from __future__ import annotations

from .fixed import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, FixedSizeChunker
from .models import TextChunk, chunk_id_for

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_OVERLAP",
    "FixedSizeChunker",
    "TextChunk",
    "chunk_id_for",
]
