"""Chunking: pages become the units the retriever actually ranks."""

from __future__ import annotations

from .models import TextChunk, chunk_id_for

__all__ = ["TextChunk", "chunk_id_for"]
