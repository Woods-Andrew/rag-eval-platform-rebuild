"""Retrieval: ranking a corpus of chunks against a query."""

from __future__ import annotations

from .base import Retriever, validate_query, validate_top_k
from .bm25 import BM25Retriever
from .corpus import Corpus
from .dense import DenseRetriever
from .encoder import DEFAULT_MODEL, SentenceTransformerEncoder, TextEncoder, l2_normalize
from .fusion import DEFAULT_RRF_K, reciprocal_rank_fusion, rrf_scores
from .hybrid import DEFAULT_CANDIDATE_MULTIPLIER, HybridRetriever
from .models import RetrievalResult, rank_results
from .tokenize import tokenize

__all__ = [
    "DEFAULT_CANDIDATE_MULTIPLIER",
    "DEFAULT_MODEL",
    "DEFAULT_RRF_K",
    "BM25Retriever",
    "Corpus",
    "DenseRetriever",
    "HybridRetriever",
    "RetrievalResult",
    "Retriever",
    "SentenceTransformerEncoder",
    "TextEncoder",
    "l2_normalize",
    "rank_results",
    "reciprocal_rank_fusion",
    "rrf_scores",
    "tokenize",
    "validate_query",
    "validate_top_k",
]
