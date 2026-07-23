"""Chunking sweeps: corpus statistics, per-variant labels, and what a sweep refuses."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rag_eval.chunking import FixedSizeChunker, StructureAwareChunker
from rag_eval.cli import main
from rag_eval.evaluation import BenchmarkError
from rag_eval.experiments import (
    CROSS_VARIANT_CAVEAT,
    ChunkingVariant,
    describe_corpus,
    format_sweep,
    parse_variant,
    run_sweep,
    write_sweep,
)
from rag_eval.pipeline import build_corpus
from rag_eval.retrieval import BM25Retriever, Corpus, Retriever

PdfFactory = Callable[..., Path]

PAGES = [
    "Introduction\nThe multi-omics embedding is disease aware and adaptive for patients.",
    "Methods\nMissing modalities are imputed with a learned prior across cohorts.",
    "Results\nThe adaptive gate improved downstream survival prediction accuracy.",
    "Discussion\nAblations show the disease aware routing contributes most of the gain.",
    "Related Work\nEarlier fusion approaches concatenate modality features naively.",
    "Conclusion\nAn agentic framework for adaptive multi-omics embedding is presented.",
    "Appendix\nHyperparameters were selected by grid search over the validation split.",
]


@pytest.fixture
def paper(make_pdf: PdfFactory) -> Path:
    return make_pdf(PAGES, name="paper.pdf")


def bm25_only(corpus: Corpus) -> dict[str, Retriever]:
    return {"bm25": BM25Retriever(corpus)}


def label_file(path: Path, corpus: Corpus, *, query: str = "imputed prior") -> Path:
    """Write a benchmark whose labels are real chunk IDs from ``corpus``.

    Labels come out of the pipeline rather than being written by hand here, which is
    the same rule the real benchmark follows: an ID that no chunk produced is not a
    label, it is a guess.
    """
    payload = {
        "document": "paper.pdf",
        "questions": [
            {
                "id": "q1",
                "query": query,
                "category": "methodology",
                "relevant_chunk_ids": [corpus.chunk_ids[1]],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestCorpusStats:
    def test_counts_chunks_and_pages(self, paper: Path) -> None:
        stats = describe_corpus(build_corpus(paper))

        assert stats.chunk_count == len(PAGES)
        assert stats.page_count == len(PAGES)

    def test_word_distribution_is_reported(self, paper: Path) -> None:
        corpus = build_corpus(paper)
        counts = sorted(len(chunk.text.split()) for chunk in corpus)
        stats = describe_corpus(corpus)

        assert stats.min_words == counts[0]
        assert stats.max_words == counts[-1]
        assert stats.total_words == sum(counts)
        assert stats.mean_words == pytest.approx(sum(counts) / len(counts))

    def test_stats_need_no_relevance_labels(self, paper: Path) -> None:
        # The half of a chunking experiment that is measurable before any labelling.
        assert describe_corpus(build_corpus(paper)).chunk_count > 0

    def test_chunkers_produce_different_shapes(self, paper: Path) -> None:
        fixed = describe_corpus(build_corpus(paper, FixedSizeChunker(chunk_size=8, overlap=2)))
        whole = describe_corpus(build_corpus(paper, FixedSizeChunker()))

        assert fixed.chunk_count > whole.chunk_count
        assert fixed.max_words <= 8


class TestSweepValidation:
    def test_a_sweep_needs_at_least_one_variant(self, paper: Path) -> None:
        with pytest.raises(ValueError, match="at least one chunking variant"):
            run_sweep(paper, [], retriever_factory=bm25_only, k=5)

    def test_non_positive_k_is_rejected(self, paper: Path) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            run_sweep(paper, [ChunkingVariant("fixed", FixedSizeChunker())],
                      retriever_factory=bm25_only, k=0)

    def test_duplicate_variant_names_are_rejected(self, paper: Path) -> None:
        variants = [
            ChunkingVariant("fixed", FixedSizeChunker(chunk_size=50)),
            ChunkingVariant("fixed", FixedSizeChunker(chunk_size=90)),
        ]

        with pytest.raises(ValueError, match="duplicate variant name"):
            run_sweep(paper, variants, retriever_factory=bm25_only, k=5)

    def test_a_variant_needs_a_name(self) -> None:
        with pytest.raises(ValueError, match="needs a name"):
            ChunkingVariant("  ", FixedSizeChunker())


class TestSweepWithoutLabels:
    def test_a_variant_without_a_benchmark_still_reports_its_shape(self, paper: Path) -> None:
        result = run_sweep(
            paper,
            [ChunkingVariant("fixed", FixedSizeChunker())],
            retriever_factory=bm25_only,
            k=5,
        )

        assert len(result) == 1
        assert result.variants[0].stats.chunk_count == len(PAGES)
        assert result.variants[0].was_scored is False

    def test_no_retriever_is_built_when_there_is_nothing_to_score(self, paper: Path) -> None:
        # Retrieval is the expensive half; an unlabelled variant must not pay for it.
        calls: list[Corpus] = []

        def factory(corpus: Corpus) -> dict[str, Retriever]:
            calls.append(corpus)
            return bm25_only(corpus)

        run_sweep(paper, [ChunkingVariant("fixed", FixedSizeChunker())],
                  retriever_factory=factory, k=5)

        assert calls == []

    def test_an_unscored_sweep_says_so_rather_than_printing_zeros(self, paper: Path) -> None:
        result = run_sweep(paper, [ChunkingVariant("fixed", FixedSizeChunker())],
                           retriever_factory=bm25_only, k=5)

        rendered = format_sweep(result)

        assert "only corpus shape was measured" in rendered
        assert "unmeasured" in rendered


class TestSweepWithLabels:
    def test_a_labelled_variant_is_scored(self, paper: Path, tmp_path: Path) -> None:
        corpus = build_corpus(paper)
        variant = ChunkingVariant(
            "fixed", FixedSizeChunker(), label_file(tmp_path / "fixed.json", corpus)
        )

        result = run_sweep(paper, [variant], retriever_factory=bm25_only, k=5)

        assert result.variants[0].was_scored is True
        assert [report.retriever_name for report in result.variants[0].reports] == ["bm25"]

    def test_each_variant_is_scored_against_its_own_corpus(
        self, paper: Path, tmp_path: Path
    ) -> None:
        # The point of the design: two variants, two label files, two corpora, and no
        # cross-contamination between them.
        fixed_corpus = build_corpus(paper, FixedSizeChunker())
        structure_corpus = build_corpus(paper, StructureAwareChunker(min_words=1))
        variants = [
            ChunkingVariant(
                "fixed", FixedSizeChunker(), label_file(tmp_path / "a.json", fixed_corpus)
            ),
            ChunkingVariant(
                "structure",
                StructureAwareChunker(min_words=1),
                label_file(tmp_path / "b.json", structure_corpus),
            ),
        ]

        result = run_sweep(paper, variants, retriever_factory=bm25_only, k=5)

        assert len(result.scored_variants) == 2
        assert result.is_comparable is True

    def test_labels_from_another_chunking_configuration_are_rejected(
        self, paper: Path, tmp_path: Path
    ) -> None:
        # Chunk IDs carry a text digest, so labels written under one chunker cannot
        # silently be scored under another. This is the guard that makes a sweep honest.
        fixed_labels = label_file(tmp_path / "fixed.json", build_corpus(paper))
        variant = ChunkingVariant(
            "structure", StructureAwareChunker(min_words=1), fixed_labels
        )

        with pytest.raises(BenchmarkError, match="not in the corpus"):
            run_sweep(paper, [variant], retriever_factory=bm25_only, k=5)

    def test_a_mixed_sweep_scores_what_it_can(self, paper: Path, tmp_path: Path) -> None:
        variants = [
            ChunkingVariant(
                "fixed", FixedSizeChunker(), label_file(tmp_path / "a.json", build_corpus(paper))
            ),
            ChunkingVariant("fixed:80/10", FixedSizeChunker(chunk_size=80, overlap=10)),
        ]

        result = run_sweep(paper, variants, retriever_factory=bm25_only, k=5)

        assert len(result.scored_variants) == 1
        assert result.is_comparable is False
        assert "unscored (no benchmark): fixed:80/10" in format_sweep(result)


class TestReporting:
    def test_the_cross_variant_caveat_appears_only_with_two_scored_variants(
        self, paper: Path, tmp_path: Path
    ) -> None:
        single = run_sweep(
            paper,
            [ChunkingVariant("fixed", FixedSizeChunker(),
                             label_file(tmp_path / "a.json", build_corpus(paper)))],
            retriever_factory=bm25_only,
            k=5,
        )

        assert CROSS_VARIANT_CAVEAT not in format_sweep(single)

    def test_two_scored_variants_carry_the_caveat(self, paper: Path, tmp_path: Path) -> None:
        structure = StructureAwareChunker(min_words=1)
        variants = [
            ChunkingVariant("fixed", FixedSizeChunker(),
                            label_file(tmp_path / "a.json", build_corpus(paper))),
            ChunkingVariant("structure", structure,
                            label_file(tmp_path / "b.json", build_corpus(paper, structure))),
        ]

        rendered = format_sweep(run_sweep(paper, variants, retriever_factory=bm25_only, k=5))

        assert CROSS_VARIANT_CAVEAT in rendered

    def test_every_variant_appears_in_the_corpus_table(self, paper: Path) -> None:
        variants = [
            ChunkingVariant("fixed:50/10", FixedSizeChunker(chunk_size=50, overlap=10)),
            ChunkingVariant("structure", StructureAwareChunker(min_words=1)),
        ]

        rendered = format_sweep(run_sweep(paper, variants, retriever_factory=bm25_only, k=5))

        assert "fixed:50/10" in rendered
        assert "structure" in rendered


class TestJsonOutput:
    def test_per_question_scores_are_written(self, paper: Path, tmp_path: Path) -> None:
        result = run_sweep(
            paper,
            [ChunkingVariant("fixed", FixedSizeChunker(),
                             label_file(tmp_path / "a.json", build_corpus(paper)))],
            retriever_factory=bm25_only,
            k=5,
        )

        written = json.loads(write_sweep(result, tmp_path / "out.json").read_text())
        retriever = written["variants"][0]["retrievers"][0]

        assert written["document"] == "paper.pdf"
        assert written["k"] == 5
        assert retriever["per_question"][0]["question_id"] == "q1"
        # The auditable part: which chunks actually came back, not just a mean.
        assert isinstance(retriever["per_question"][0]["retrieved_chunk_ids"], list)

    def test_missing_directories_are_created(self, paper: Path, tmp_path: Path) -> None:
        result = run_sweep(paper, [ChunkingVariant("fixed", FixedSizeChunker())],
                           retriever_factory=bm25_only, k=5)

        written = write_sweep(result, tmp_path / "results" / "nested" / "out.json")

        assert written.is_file()

    def test_an_unscored_variant_serializes_without_pretending_to_have_scores(
        self, paper: Path, tmp_path: Path
    ) -> None:
        result = run_sweep(paper, [ChunkingVariant("fixed", FixedSizeChunker())],
                           retriever_factory=bm25_only, k=5)

        written = json.loads(write_sweep(result, tmp_path / "out.json").read_text())

        assert written["variants"][0]["scored"] is False
        assert written["variants"][0]["retrievers"] == []
        assert written["variants"][0]["stats"]["chunk_count"] == len(PAGES)


class TestVariantSpec:
    def test_bare_strategies_use_the_defaults(self) -> None:
        assert parse_variant("fixed").chunker == FixedSizeChunker()
        assert parse_variant("structure").chunker == StructureAwareChunker()

    def test_parameters_are_parsed(self) -> None:
        assert parse_variant("fixed:300/60").chunker == FixedSizeChunker(
            chunk_size=300, overlap=60
        )
        assert parse_variant("structure:150/20").chunker == StructureAwareChunker(
            max_words=150, min_words=20
        )

    def test_the_variant_is_named_after_its_spec(self, paper: Path, tmp_path: Path) -> None:
        labels = label_file(tmp_path / "a.json", build_corpus(paper))

        assert parse_variant(f"fixed:300/60={labels}").name == "fixed:300/60"

    def test_a_benchmark_path_is_attached(self, paper: Path, tmp_path: Path) -> None:
        labels = label_file(tmp_path / "a.json", build_corpus(paper))

        assert parse_variant(f"fixed={labels}").benchmark_path == labels

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ("nope", "unknown chunking strategy"),
            ("fixed:300", "needs both parameters"),
            ("fixed:a/b", "non-integer parameters"),
            ("fixed=", "no benchmark path"),
            ("fixed=nowhere.json", "no benchmark file"),
            ("", "empty chunking variant"),
        ],
    )
    def test_malformed_specs_are_rejected(self, spec: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_variant(spec)

    def test_invalid_chunker_parameters_surface_the_chunker_error(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            parse_variant("fixed:50/50")


class TestSweepCommand:
    def test_a_stats_only_sweep_runs_without_any_benchmark(
        self, paper: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["sweep", str(paper), "-v", "fixed", "-v", "structure"])

        assert exit_code == 0
        assert "only corpus shape was measured" in capsys.readouterr().out

    def test_a_labelled_sweep_prints_scores(
        self, paper: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        labels = label_file(tmp_path / "a.json", build_corpus(paper))

        exit_code = main(["sweep", str(paper), "-v", f"fixed={labels}", "-r", "bm25"])

        assert exit_code == 0
        assert "Recall@K" in capsys.readouterr().out

    def test_results_are_written_when_asked(
        self, paper: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        destination = tmp_path / "results" / "sweep.json"

        main(["sweep", str(paper), "-v", "fixed", "-o", str(destination)])

        assert destination.is_file()
        assert "wrote" in capsys.readouterr().out

    def test_a_bad_variant_spec_exits_nonzero(
        self, paper: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(["sweep", str(paper), "-v", "nonsense"])

        assert exit_code == 1
        assert "unknown chunking strategy" in capsys.readouterr().err

    def test_mismatched_labels_exit_nonzero(
        self, paper: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        labels = label_file(tmp_path / "a.json", build_corpus(paper))

        exit_code = main(["sweep", str(paper), "-v", f"structure={labels}", "-r", "bm25"])

        assert exit_code == 1
        assert "not in the corpus" in capsys.readouterr().err
