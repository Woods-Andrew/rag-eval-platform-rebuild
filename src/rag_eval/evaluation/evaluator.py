"""Running a benchmark against a retriever."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..retrieval import Retriever
from .metrics import ndcg_at_k, recall_at_k
from .models import BenchmarkQuestion, EvaluationReport, QueryScore

__all__ = ["compare", "evaluate"]


def evaluate(
    retriever: Retriever,
    questions: Sequence[BenchmarkQuestion],
    *,
    k: int,
    name: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> EvaluationReport:
    """Score ``retriever`` on every question at cutoff ``k``.

    The retriever is only ever used through the ``Retriever`` protocol, so lexical,
    dense, fused, and reranked strategies are all measured by the same code path.
    Nothing here imports generation: retrieval quality is measured on its own terms.

    Retrieval runs once per question at ``top_k=k`` and both metrics read that one
    ranking, so they can never disagree about what was retrieved.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not questions:
        raise ValueError("cannot evaluate against an empty question set")
    if not isinstance(retriever, Retriever):
        raise TypeError("retriever must expose a callable retrieve(query, *, top_k)")

    scores = [
        _score_question(retriever, question, k=k) for question in questions
    ]
    return EvaluationReport(
        retriever_name=name or type(retriever).__name__,
        k=k,
        scores=tuple(scores),
        metadata=metadata or {},
    )


def _score_question(
    retriever: Retriever, question: BenchmarkQuestion, *, k: int
) -> QueryScore:
    retrieved = tuple(result.chunk_id for result in retriever.retrieve(question.query, top_k=k))
    return QueryScore(
        question_id=question.question_id,
        category=question.category,
        k=k,
        recall=recall_at_k(retrieved, question.relevant_chunk_ids, k),
        ndcg=ndcg_at_k(retrieved, question.relevant_chunk_ids, k),
        retrieved_chunk_ids=retrieved,
    )


def compare(
    retrievers: Mapping[str, Retriever],
    questions: Sequence[BenchmarkQuestion],
    *,
    k: int,
) -> list[EvaluationReport]:
    """Evaluate several retrievers on the same questions, in the order given.

    Reports come back in insertion order, never sorted by score. Ordering the output by
    performance invites reading the first row as "the winner"; what the comparison
    means is the caller's call, and a simpler method beating a fancier one is a result
    to report, not to hide.
    """
    if not retrievers:
        raise ValueError("cannot compare an empty set of retrievers")

    return [evaluate(retriever, questions, k=k, name=name) for name, retriever in retrievers.items()]
