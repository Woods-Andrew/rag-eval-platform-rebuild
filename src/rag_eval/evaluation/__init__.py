"""Evaluation: measuring retrieval quality against hand-labelled judgments."""

from __future__ import annotations

from .benchmark import Benchmark, BenchmarkError, load_benchmark
from .evaluator import compare, evaluate
from .metrics import dcg, ndcg_at_k, recall_at_k
from .models import BenchmarkQuestion, EvaluationReport, QueryScore

__all__ = [
    "Benchmark",
    "BenchmarkError",
    "BenchmarkQuestion",
    "EvaluationReport",
    "QueryScore",
    "compare",
    "dcg",
    "evaluate",
    "load_benchmark",
    "ndcg_at_k",
    "recall_at_k",
]
