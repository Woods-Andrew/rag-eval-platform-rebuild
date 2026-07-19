"""Benchmark loading, corpus validation, the pipeline, and the CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rag_eval.chunking import StructureAwareChunker
from rag_eval.cli import main
from rag_eval.evaluation import BenchmarkError, load_benchmark
from rag_eval.pipeline import build_corpus

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
    """A multi-page PDF. Size matters: Okapi IDF degenerates on a tiny corpus."""
    return make_pdf(PAGES, name="paper.pdf")


def write_benchmark(path: Path, questions: list[dict[str, object]], **top: object) -> Path:
    payload = {"document": "paper.pdf", "chunking": "fixed(200/40)", "questions": questions, **top}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestPipeline:
    def test_builds_a_corpus_from_a_pdf(self, paper: Path) -> None:
        corpus = build_corpus(paper)

        assert len(corpus) == len(PAGES)
        assert corpus.sources == ("paper.pdf",)

    def test_the_chunker_is_swappable(self, paper: Path) -> None:
        structure = build_corpus(paper, StructureAwareChunker(min_words=1))

        assert any(chunk.section for chunk in structure)

    def test_changing_the_chunker_changes_every_chunk_id(self, paper: Path) -> None:
        # This is why relevance labels are tied to one chunking configuration.
        fixed_ids = set(build_corpus(paper).chunk_ids)
        structure_ids = set(build_corpus(paper, StructureAwareChunker(min_words=1)).chunk_ids)

        assert fixed_ids.isdisjoint(structure_ids)

    def test_a_missing_pdf_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_corpus(tmp_path / "absent.pdf")


class TestLoadBenchmark:
    def test_loads_questions_with_their_metadata(self, tmp_path: Path, paper: Path) -> None:
        corpus = build_corpus(paper)
        chunk_id = corpus.chunk_ids[1]
        path = write_benchmark(
            tmp_path / "b.json",
            [
                {
                    "id": "q1",
                    "query": "imputed missing modalities",
                    "category": "lexical",
                    "relevant_chunk_ids": [chunk_id],
                }
            ],
        )

        benchmark = load_benchmark(path, corpus)

        assert len(benchmark) == 1
        assert benchmark.questions[0].query == "imputed missing modalities"
        assert benchmark.categories == ("lexical",)
        assert benchmark.label_count == 1
        assert benchmark.document == "paper.pdf"

    def test_a_label_that_matches_no_chunk_is_an_error(
        self, tmp_path: Path, paper: Path
    ) -> None:
        # The integrity guard: a dangling ID means the document or the chunking
        # settings changed since labelling, so scoring would measure the wrong thing.
        path = write_benchmark(
            tmp_path / "b.json",
            [{"id": "q1", "query": "anything", "relevant_chunk_ids": ["paper-p002-c00-deadbeef"]}],
        )

        with pytest.raises(BenchmarkError, match="not in the corpus"):
            load_benchmark(path, build_corpus(paper))

    def test_the_corpus_check_is_skipped_when_no_corpus_is_given(self, tmp_path: Path) -> None:
        path = write_benchmark(
            tmp_path / "b.json",
            [{"id": "q1", "query": "anything", "relevant_chunk_ids": ["whatever"]}],
        )

        assert len(load_benchmark(path)) == 1

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No benchmark file"):
            load_benchmark(tmp_path / "absent.json")

    def test_malformed_json_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(BenchmarkError, match="not valid JSON"):
            load_benchmark(path)

    def test_an_empty_question_list_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(BenchmarkError, match="non-empty 'questions' array"):
            load_benchmark(write_benchmark(tmp_path / "b.json", []))

    def test_duplicate_question_ids_are_rejected(self, tmp_path: Path) -> None:
        path = write_benchmark(
            tmp_path / "b.json",
            [
                {"id": "q1", "query": "a", "relevant_chunk_ids": ["x"]},
                {"id": "q1", "query": "b", "relevant_chunk_ids": ["y"]},
            ],
        )

        with pytest.raises(BenchmarkError, match="duplicate question id"):
            load_benchmark(path)

    def test_a_question_with_no_labels_is_rejected(self, tmp_path: Path) -> None:
        path = write_benchmark(
            tmp_path / "b.json", [{"id": "q1", "query": "a", "relevant_chunk_ids": []}]
        )

        with pytest.raises(BenchmarkError, match="at least one relevant chunk"):
            load_benchmark(path)

    def test_a_missing_id_is_rejected(self, tmp_path: Path) -> None:
        path = write_benchmark(
            tmp_path / "b.json", [{"query": "a", "relevant_chunk_ids": ["x"]}]
        )

        with pytest.raises(BenchmarkError, match="has no 'id'"):
            load_benchmark(path)

    def test_labels_of_the_wrong_type_are_rejected(self, tmp_path: Path) -> None:
        path = write_benchmark(
            tmp_path / "b.json", [{"id": "q1", "query": "a", "relevant_chunk_ids": "x"}]
        )

        with pytest.raises(BenchmarkError, match="list of strings"):
            load_benchmark(path)


class TestCLI:
    def test_index_reports_corpus_statistics(
        self, paper: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["index", str(paper)]) == 0

        output = capsys.readouterr().out
        assert f"{len(PAGES)} chunks" in output
        assert "paper-p001-c00-" in output

    def test_index_accepts_a_chunker_choice(
        self, paper: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["index", str(paper), "--chunker", "structure"]) == 0
        assert "chunker: structure" in capsys.readouterr().out

    def test_search_prints_ranked_results(
        self, paper: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["search", str(paper), "imputed missing modalities", "-k", "3"]) == 0

        output = capsys.readouterr().out
        assert "paper.pdf p.2" in output
        assert "1. [" in output

    def test_search_reports_no_results_without_failing(
        self, paper: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["search", str(paper), "quantum chromodynamics tractor"]) == 0
        assert "no results" in capsys.readouterr().out

    def test_evaluate_prints_a_report(
        self, paper: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        corpus = build_corpus(paper)
        methods_chunk = next(c.chunk_id for c in corpus if c.page_number == 2)
        benchmark = write_benchmark(
            tmp_path / "b.json",
            [
                {
                    "id": "q1",
                    "query": "imputed missing modalities",
                    "category": "lexical",
                    "relevant_chunk_ids": [methods_chunk],
                }
            ],
        )

        assert main(["evaluate", str(paper), str(benchmark), "-k", "3", "-r", "bm25"]) == 0

        output = capsys.readouterr().out
        assert "Recall@K" in output
        assert "bm25" in output

    def test_a_missing_pdf_exits_nonzero_with_a_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["index", str(tmp_path / "absent.pdf")]) == 1
        assert "error:" in capsys.readouterr().err

    def test_a_broken_benchmark_exits_nonzero(
        self, paper: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        benchmark = write_benchmark(
            tmp_path / "b.json",
            [{"id": "q1", "query": "a", "relevant_chunk_ids": ["paper-p001-c00-deadbeef"]}],
        )

        assert main(["evaluate", str(paper), str(benchmark), "-r", "bm25"]) == 1
        assert "not in the corpus" in capsys.readouterr().err

    def test_the_cli_never_loads_an_embedding_model_for_bm25(
        self, paper: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The dense import lives inside _make_retriever, so a bm25 run stays offline.
        import rag_eval.cli as cli_module

        def explode() -> None:
            raise AssertionError("bm25 must not construct an encoder")

        monkeypatch.setattr(cli_module, "DenseRetriever", explode)

        assert main(["search", str(paper), "imputed missing modalities"]) == 0
