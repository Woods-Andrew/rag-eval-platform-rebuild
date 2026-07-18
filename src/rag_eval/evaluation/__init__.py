"""Evaluation: measuring retrieval quality against hand-labelled judgments."""

from __future__ import annotations

from .evaluator import compare, evaluate
from .metrics import dcg, ndcg_at_k, recall_at_k
from .models import BenchmarkQuestion, EvaluationReport, QueryScore

__all__ = [
    "BenchmarkQuestion",
    "EvaluationReport",
    "QueryScore",
    "compare",
    "dcg",
    "evaluate",
    "ndcg_at_k",
    "recall_at_k",
]
