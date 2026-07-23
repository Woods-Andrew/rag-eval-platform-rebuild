"""Experiments: sweeping a document across chunking configurations.

Imports nothing from ``generation`` and nothing from the UI. A sweep measures
retrieval, and a chunking strategy is measured the same way any other variable is —
by re-indexing and re-scoring, never by argument.
"""

from __future__ import annotations

from .report import CROSS_VARIANT_CAVEAT, format_sweep, write_sweep
from .spec import SPEC_HELP, parse_variant
from .stats import CorpusStats, describe_corpus
from .sweep import ChunkingVariant, SweepResult, VariantResult, run_sweep

__all__ = [
    "CROSS_VARIANT_CAVEAT",
    "SPEC_HELP",
    "ChunkingVariant",
    "CorpusStats",
    "SweepResult",
    "VariantResult",
    "describe_corpus",
    "format_sweep",
    "parse_variant",
    "run_sweep",
    "write_sweep",
]
