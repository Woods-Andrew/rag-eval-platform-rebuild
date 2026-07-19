"""PDF to searchable corpus, in one place.

The ingestion → chunking → corpus path is identical for the CLI, the benchmark, and
(later) the UI. Keeping it here means all three index a document exactly the same way,
which is what makes a chunk ID stable enough for a relevance label to point at.
"""

from __future__ import annotations

from pathlib import Path

from .chunking import Chunker, FixedSizeChunker
from .ingestion import load_pdf
from .retrieval import Corpus

__all__ = ["build_corpus"]


def build_corpus(pdf_path: str | Path, chunker: Chunker | None = None) -> Corpus:
    """Load a PDF and chunk it into a searchable corpus.

    Defaults to fixed-size chunking, which is the baseline every other strategy is
    measured against. Pass a different chunker to index the same document differently —
    but note that changing the chunker changes every chunk ID, and therefore invalidates
    any relevance labels written against the previous run.
    """
    pages = load_pdf(pdf_path)
    chunks = (chunker or FixedSizeChunker()).chunk_pages(pages)
    if not chunks:
        raise ValueError(f"{Path(pdf_path).name} produced no chunks")
    return Corpus(chunks)
