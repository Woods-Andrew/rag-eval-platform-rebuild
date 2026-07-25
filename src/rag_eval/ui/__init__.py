"""UI: the Streamlit interface and the service layer beneath it.

Importing this package deliberately does **not** import Streamlit. Only ``ui.app``
does, and nothing else in the project imports ``ui.app`` — the interface is a consumer
of retrieval, never a dependency of it.
"""

from __future__ import annotations

from .service import CHUNKERS, RETRIEVERS, RetrievalService, SearchOutcome, make_chunker

__all__ = [
    "CHUNKERS",
    "RETRIEVERS",
    "RetrievalService",
    "SearchOutcome",
    "make_chunker",
]
