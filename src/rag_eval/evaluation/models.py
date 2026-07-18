"""The benchmark question, and the shape of an evaluation run's output."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = ["BenchmarkQuestion", "EvaluationReport", "QueryScore"]


@dataclass(frozen=True)
class BenchmarkQuestion:
    """One labelled question: the query, and the chunks known to answer it.

    ``category`` records the kind of retrieval the question stresses — lexical,
    paraphrase, acronym, methodology, numeric, finding — so a report can show *where*
    a retriever wins rather than only that it does.

    Every ID in ``relevant_chunk_ids`` must correspond to a chunk the pipeline actually
    produced; that is checked against the corpus when the benchmark is loaded, not
    here, because a question does not know what corpus it will be run against.
    """

    question_id: str
    query: str
    relevant_chunk_ids: frozenset[str]
    category: str = "uncategorized"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must not be empty")
        if not self.query.strip():
            raise ValueError(f"{self.question_id}: query must not be empty")
        if not self.relevant_chunk_ids:
            raise ValueError(
                f"{self.question_id}: a benchmark question needs at least one relevant "
                "chunk; an unanswerable question cannot be scored"
            )
        object.__setattr__(self, "relevant_chunk_ids", frozenset(self.relevant_chunk_ids))


@dataclass(frozen=True)
class QueryScore:
    """One question's metrics under one retriever."""

    question_id: str
    category: str
    k: int
    recall: float
    ndcg: float
    retrieved_chunk_ids: tuple[str, ...] = ()

    @property
    def found_any(self) -> bool:
        """True when at least one relevant chunk made the top K."""
        return self.recall > 0.0


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated results for one retriever at one cutoff.

    Means are unweighted across questions: every question counts once regardless of how
    many relevant chunks it has, so a single heavily-labelled question cannot dominate
    the headline number.
    """

    retriever_name: str
    k: int
    scores: tuple[QueryScore, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("an evaluation report needs at least one scored question")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def mean_recall(self) -> float:
        return _mean(score.recall for score in self.scores)

    @property
    def mean_ndcg(self) -> float:
        return _mean(score.ndcg for score in self.scores)

    @property
    def questions_with_no_hit(self) -> tuple[str, ...]:
        """Questions where nothing relevant reached the top K — the outright failures."""
        return tuple(score.question_id for score in self.scores if not score.found_any)

    def by_category(self) -> dict[str, tuple[float, float]]:
        """Mean ``(recall, nDCG)`` per question category, for diagnosing *where* it wins."""
        grouped: dict[str, list[QueryScore]] = {}
        for score in self.scores:
            grouped.setdefault(score.category, []).append(score)
        return {
            category: (
                _mean(score.recall for score in scores),
                _mean(score.ndcg for score in scores),
            )
            for category, scores in sorted(grouped.items())
        }


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0
