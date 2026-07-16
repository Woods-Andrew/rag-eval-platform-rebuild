"""Retrieval: ranking a corpus of chunks against a query."""

from __future__ import annotations

from .base import Retriever, validate_query, validate_top_k
from .bm25 import BM25Retriever
from .corpus import Corpus
from .dense import DenseRetriever
from .encoder import DEFAULT_MODEL, SentenceTransformerEncoder, TextEncoder, l2_normalize
from .models import RetrievalResult, rank_results
from .tokenize import tokenize

__all__ = [
    "DEFAULT_MODEL",
    "BM25Retriever",
    "Corpus",
    "DenseRetriever",
    "RetrievalResult",
    "Retriever",
    "SentenceTransformerEncoder",
    "TextEncoder",
    "l2_normalize",
    "rank_results",
    "tokenize",
    "validate_query",
    "validate_top_k",
]
