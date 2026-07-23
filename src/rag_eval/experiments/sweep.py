"""Running the same benchmark across several chunking configurations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..chunking import Chunker
from ..evaluation import Benchmark, EvaluationReport, compare, load_benchmark
from ..pipeline import build_corpus
from ..retrieval import Corpus, Retriever
from .stats import CorpusStats, describe_corpus

__all__ = ["ChunkingVariant", "SweepResult", "VariantResult", "run_sweep"]

RetrieverFactory = Callable[[Corpus], Mapping[str, Retriever]]


@dataclass(frozen=True)
class ChunkingVariant:
    """One chunking configuration, and the labels written against it.

    ``benchmark_path`` is per-variant and that is not an inconvenience to design around
    — it is the honest consequence of how chunk IDs work. An ID contains a digest of
    its chunk's text, so re-chunking a document invalidates every label written against
    the previous run. A sweep that reused one label file across variants would be
    scoring most of them against IDs that no longer resolve.

    A variant with no benchmark still contributes its corpus statistics, which is what
    makes a sweep runnable before any labelling has happened.
    """

    name: str
    chunker: Chunker
    benchmark_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a chunking variant needs a name")


@dataclass(frozen=True)
class VariantResult:
    """What one chunking configuration produced: a corpus shape, and maybe scores."""

    name: str
    chunker: str
    stats: CorpusStats
    reports: tuple[EvaluationReport, ...] = ()
    benchmark: Benchmark | None = None

    @property
    def was_scored(self) -> bool:
        """False when this variant had no benchmark, so only its shape was measured."""
        return bool(self.reports)

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.name,
            "chunker": self.chunker,
            "stats": self.stats.to_dict(),
            "scored": self.was_scored,
            "questions": len(self.benchmark) if self.benchmark else 0,
            "retrievers": [
                {
                    "name": report.retriever_name,
                    "k": report.k,
                    "mean_recall": report.mean_recall,
                    "mean_ndcg": report.mean_ndcg,
                    "questions_with_no_hit": list(report.questions_with_no_hit),
                    "by_category": {
                        category: {"recall": recall, "ndcg": ndcg}
                        for category, (recall, ndcg) in report.by_category().items()
                    },
                    "per_question": [
                        {
                            "question_id": score.question_id,
                            "category": score.category,
                            "recall": score.recall,
                            "ndcg": score.ndcg,
                            "retrieved_chunk_ids": list(score.retrieved_chunk_ids),
                        }
                        for score in report.scores
                    ],
                }
                for report in self.reports
            ],
        }


@dataclass(frozen=True)
class SweepResult:
    """Every variant's outcome for one document at one cutoff."""

    document: str
    k: int
    variants: tuple[VariantResult, ...]

    def __len__(self) -> int:
        return len(self.variants)

    @property
    def scored_variants(self) -> tuple[VariantResult, ...]:
        return tuple(variant for variant in self.variants if variant.was_scored)

    @property
    def is_comparable(self) -> bool:
        """True when more than one variant carries scores worth putting side by side."""
        return len(self.scored_variants) > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "document": self.document,
            "k": self.k,
            "comparable": self.is_comparable,
            "variants": [variant.to_dict() for variant in self.variants],
        }


def run_sweep(
    pdf_path: str | Path,
    variants: Sequence[ChunkingVariant],
    *,
    retriever_factory: RetrieverFactory,
    k: int,
) -> SweepResult:
    """Chunk one document several ways and benchmark each result.

    ``retriever_factory`` builds the retrievers to measure for a given corpus. It is
    injected rather than chosen here for the usual reason — the real one constructs
    embedding and cross-encoder models, and no unit test should have to download those
    to check that a sweep aggregates correctly.

    Every variant re-chunks and re-indexes from the same PDF, because that is the only
    way the comparison is fair: the corpus a variant is scored on has to be the corpus
    that variant actually produces.

    Raises:
        ValueError: no variants, duplicate variant names, or a non-positive ``k``.
        BenchmarkError: a variant's labels do not resolve against its own corpus, which
            means they were written under different chunking settings.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not variants:
        raise ValueError("a sweep needs at least one chunking variant")

    names = [variant.name for variant in variants]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate variant name(s): {', '.join(duplicates)}")

    document = Path(pdf_path).name
    results = [
        _run_variant(pdf_path, variant, retriever_factory=retriever_factory, k=k)
        for variant in variants
    ]
    return SweepResult(document=document, k=k, variants=tuple(results))


def _run_variant(
    pdf_path: str | Path,
    variant: ChunkingVariant,
    *,
    retriever_factory: RetrieverFactory,
    k: int,
) -> VariantResult:
    corpus = build_corpus(pdf_path, variant.chunker)
    stats = describe_corpus(corpus)
    chunker = type(variant.chunker).__name__

    if variant.benchmark_path is None:
        return VariantResult(name=variant.name, chunker=chunker, stats=stats)

    benchmark = load_benchmark(variant.benchmark_path, corpus)
    reports = compare(retriever_factory(corpus), list(benchmark.questions), k=k)
    return VariantResult(
        name=variant.name,
        chunker=chunker,
        stats=stats,
        reports=tuple(reports),
        benchmark=benchmark,
    )
