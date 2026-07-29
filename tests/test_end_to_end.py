"""End to end: a PDF goes in, ranked evidence and a cited answer come out.

Every other test file checks one stage in isolation. These run the real path — PDF →
pages → chunks → corpus → retrieval → fusion → reranking → evaluation → generation —
with only the ML models faked, because a pipeline can be correct at every seam and
still be wrong when assembled.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from rag_eval.chunking import FixedSizeChunker, StructureAwareChunker
from rag_eval.cli import main
from rag_eval.evaluation import BenchmarkQuestion, compare, evaluate, load_benchmark
from rag_eval.factory import RetrieverFactory
from rag_eval.generation import AnswerGenerator
from rag_eval.pipeline import build_corpus
from rag_eval.retrieval import Corpus, EmbeddingCache
from tests.fakes import FakeEncoder, FakeLanguageModel, FakeReranker

PdfFactory = Callable[..., Path]

PAGES = [
    "Introduction\n"
    "The multi-omics embedding is disease aware and adaptive for patients across cohorts.",
    "Methods\n"
    "Missing modalities are imputed with a learned prior estimated from the training split.",
    "Results\n"
    "The adaptive gate improved downstream survival prediction accuracy by four points.",
    "Discussion\n"
    "Ablations show the disease aware routing contributes most of the observed gain.",
    "Related Work\n"
    "Earlier fusion approaches concatenate modality features naively before training.",
    "Conclusion\n"
    "An agentic framework for adaptive multi-omics embedding is presented and evaluated.",
    "Appendix\n"
    "Hyperparameters were selected by grid search over the held out validation split.",
]


@pytest.fixture
def paper(make_pdf: PdfFactory) -> Path:
    return make_pdf(PAGES, name="omics.pdf")


def encoder_for(corpus: Corpus, *, query: str, favours: int) -> FakeEncoder:
    """An encoder that points ``query`` at exactly one chunk, everything else away.

    Fixed vectors rather than a real model, so the ranking a test asserts on is a
    property of the retrieval code and not of a downloaded checkpoint.
    """
    vectors: dict[str, list[float]] = {query: [1.0, 0.0]}
    for index, chunk in enumerate(corpus):
        vectors[chunk.text] = [1.0, 0.0] if index == favours else [0.0, 1.0]
    return FakeEncoder(vectors, dimension=2)


def factory_for(corpus: Corpus, encoder: FakeEncoder, **kwargs: object) -> RetrieverFactory:
    return RetrieverFactory(
        corpus,
        encoder_factory=lambda: encoder,
        reranker_factory=lambda: FakeReranker({}),
        **kwargs,  # type: ignore[arg-type]
    )


class TestPipelineIntegrity:
    def test_a_pdf_becomes_a_searchable_corpus(self, paper: Path) -> None:
        corpus = build_corpus(paper)

        assert len(corpus) == len(PAGES)
        assert corpus.sources == ("omics.pdf",)

    def test_provenance_survives_every_stage(self, paper: Path) -> None:
        # The rule that has to hold from extraction all the way to a citation.
        corpus = build_corpus(paper)
        encoder = encoder_for(corpus, query="imputed prior", favours=1)
        factory = factory_for(corpus, encoder)

        result = factory.get("hybrid").retrieve("imputed prior", top_k=1)[0]
        answer = AnswerGenerator(FakeLanguageModel("A learned prior [1].")).answer(
            "imputed prior", [result]
        )

        assert result.chunk.source == "omics.pdf"
        assert answer.citations[0].chunk_id == result.chunk_id
        assert answer.citations[0].source == "omics.pdf"
        assert answer.citations[0].page_number == result.chunk.page_number

    def test_every_strategy_returns_ranked_results_over_a_real_document(
        self, paper: Path
    ) -> None:
        corpus = build_corpus(paper)
        factory = factory_for(corpus, encoder_for(corpus, query="imputed prior", favours=1))

        for name in ("bm25", "dense", "hybrid", "reranked"):
            results = factory.get(name).retrieve("imputed prior", top_k=3)

            assert results, f"{name} returned nothing"
            assert [r.rank for r in results] == list(range(1, len(results) + 1)), name

    def test_chunk_ids_are_stable_across_identical_runs(self, paper: Path) -> None:
        # A label written today has to still resolve tomorrow.
        assert build_corpus(paper).chunk_ids == build_corpus(paper).chunk_ids

    def test_re_chunking_invalidates_every_chunk_id(self, paper: Path) -> None:
        fixed = set(build_corpus(paper, FixedSizeChunker()).chunk_ids)
        structure = set(build_corpus(paper, StructureAwareChunker(min_words=1)).chunk_ids)

        assert fixed.isdisjoint(structure)


class TestEvaluationEndToEnd:
    def test_a_labelled_benchmark_scores_a_real_corpus(self, paper: Path) -> None:
        corpus = build_corpus(paper)
        target = corpus.chunk_ids[1]
        questions = [
            BenchmarkQuestion(
                question_id="q1",
                query="imputed prior",
                relevant_chunk_ids=frozenset({target}),
                category="methodology",
            )
        ]

        factory = factory_for(corpus, encoder_for(corpus, query="imputed prior", favours=1))
        report = evaluate(factory.get("bm25"), questions, k=5)

        assert report.mean_recall == pytest.approx(1.0)
        assert report.questions_with_no_hit == ()

    def test_all_four_strategies_are_comparable_on_one_question_set(
        self, paper: Path
    ) -> None:
        corpus = build_corpus(paper)
        factory = factory_for(corpus, encoder_for(corpus, query="imputed prior", favours=1))
        questions = [
            BenchmarkQuestion(
                question_id="q1",
                query="imputed prior",
                relevant_chunk_ids=frozenset({corpus.chunk_ids[1]}),
            )
        ]

        reports = compare(factory.all(), questions, k=5)

        assert [report.retriever_name for report in reports] == [
            "bm25",
            "dense",
            "hybrid",
            "reranked",
        ]
        assert all(0.0 <= report.mean_ndcg <= 1.0 for report in reports)

    def test_a_benchmark_written_against_the_corpus_loads(
        self, paper: Path, tmp_path: Path
    ) -> None:
        corpus = build_corpus(paper)
        path = tmp_path / "benchmark.json"
        path.write_text(
            json.dumps(
                {
                    "document": "omics.pdf",
                    "questions": [
                        {
                            "id": "q1",
                            "query": "imputed prior",
                            "relevant_chunk_ids": [corpus.chunk_ids[1]],
                        }
                    ],
                }
            )
        )

        benchmark = load_benchmark(path, corpus)

        assert len(benchmark) == 1
        assert benchmark.label_count == 1


class TestGenerationEndToEnd:
    def test_a_question_becomes_a_cited_answer(self, paper: Path) -> None:
        corpus = build_corpus(paper)
        factory = factory_for(corpus, encoder_for(corpus, query="imputed prior", favours=1))
        results = factory.get("hybrid").retrieve("imputed prior", top_k=3)

        answer = AnswerGenerator(
            FakeLanguageModel("Imputed with a learned prior [1].")
        ).answer("how are missing modalities handled?", results)

        assert answer.is_grounded is True
        assert answer.cited_pages[0][0] == "omics.pdf"
        assert "sources:" in answer.format()

    def test_a_query_matching_nothing_produces_a_refusal_not_an_invention(
        self, paper: Path
    ) -> None:
        # The end-to-end version of the guarantee: no evidence, no answer, no model call.
        corpus = build_corpus(paper)
        model = FakeLanguageModel("This should never be returned.")
        results = RetrieverFactory(corpus).get("bm25").retrieve("zzzznonexistent", top_k=5)

        answer = AnswerGenerator(model).answer("zzzznonexistent", results)

        assert results == []
        assert model.calls == []
        assert answer.has_answer is False


class TestCliEndToEnd:
    def test_index_search_and_sweep_run_against_one_document(
        self, paper: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["index", str(paper)]) == 0
        assert main(["search", str(paper), "imputed prior", "-r", "bm25"]) == 0
        assert main(["sweep", str(paper), "-v", "fixed", "-v", "structure"]) == 0

        output = capsys.readouterr().out
        assert "omics.pdf: 7 chunks" in output
        assert "result(s) for 'imputed prior'" in output

    def test_evaluate_runs_a_benchmark_written_from_the_index(
        self, paper: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        corpus = build_corpus(paper)
        benchmark = tmp_path / "benchmark.json"
        benchmark.write_text(
            json.dumps(
                {
                    "document": "omics.pdf",
                    "questions": [
                        {
                            "id": "q1",
                            "query": "imputed prior",
                            "category": "methodology",
                            "relevant_chunk_ids": [corpus.chunk_ids[1]],
                        }
                    ],
                }
            )
        )

        exit_code = main(
            ["evaluate", str(paper), str(benchmark), "-r", "bm25", "-k", "5"]
        )

        assert exit_code == 0
        assert "Recall@K" in capsys.readouterr().out

    def test_a_second_run_reads_embeddings_from_the_cache(
        self, paper: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two CLI invocations, one embedding pass — the cross-run promise, end to end.
        import rag_eval.factory as factory_module

        corpus = build_corpus(paper)
        encoders = [
            encoder_for(corpus, query="imputed prior", favours=1),
            encoder_for(corpus, query="imputed prior", favours=1),
        ]
        handed_out: list[FakeEncoder] = []

        def next_encoder() -> FakeEncoder:
            encoder = encoders[len(handed_out)]
            encoder.model_name = "fake-model"  # type: ignore[attr-defined]
            handed_out.append(encoder)
            return encoder

        monkeypatch.setattr(factory_module, "_default_encoder", next_encoder)
        arguments = [
            "search", str(paper), "imputed prior",
            "-r", "dense", "--cache-dir", str(tmp_path / "cache"),
        ]

        assert main(arguments) == 0
        assert main(arguments) == 0

        assert len(encoders[0].calls) == 2  # corpus, then query
        assert encoders[1].encoded_texts == ["imputed prior"]  # query only

    def test_no_cache_forces_a_fresh_embedding_pass(
        self, paper: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import rag_eval.factory as factory_module

        corpus = build_corpus(paper)
        encoders = [encoder_for(corpus, query="q", favours=1) for _ in range(2)]
        handed_out: list[FakeEncoder] = []

        def next_encoder() -> FakeEncoder:
            encoder = encoders[len(handed_out)]
            handed_out.append(encoder)
            return encoder

        monkeypatch.setattr(factory_module, "_default_encoder", next_encoder)
        arguments = [
            "search", str(paper), "q", "-r", "dense",
            "--cache-dir", str(tmp_path / "cache"), "--no-cache",
        ]

        main(arguments)
        main(arguments)

        assert not (tmp_path / "cache").exists()
        assert len(encoders[1].calls) == 2  # corpus embedded again

    def test_ask_runs_the_whole_path_from_pdf_to_cited_answer(
        self, paper: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        from tests.test_generation import TestClaudeClient

        TestClaudeClient._stub_response(
            monkeypatch,
            {"content": [{"type": "text", "text": "Imputed with a learned prior [1]."}]},
        )

        exit_code = main(["ask", str(paper), "imputed prior", "-r", "bm25"])

        output = capsys.readouterr().out
        assert exit_code == 0
        assert "sources:" in output
        assert "omics.pdf p." in output


class TestUiEndToEnd:
    def test_the_service_indexes_searches_and_answers(self, paper: Path) -> None:
        from rag_eval.ui import RetrievalService

        corpus = build_corpus(paper)
        service = RetrievalService(
            paper,
            encoder_factory=lambda: encoder_for(corpus, query="imputed prior", favours=1),
            reranker_factory=lambda: FakeReranker({}),
            model_factory=lambda: FakeLanguageModel("A learned prior [1]."),
        )

        outcome = service.ask("imputed prior", retriever="hybrid", top_k=3)

        assert outcome.found_anything is True
        assert outcome.answer is not None
        assert outcome.answer.is_grounded is True

    def test_the_service_reuses_a_warm_embedding_cache(
        self, paper: Path, tmp_path: Path
    ) -> None:
        from rag_eval.ui import RetrievalService

        corpus = build_corpus(paper)
        cache = EmbeddingCache(tmp_path / "cache")
        first = encoder_for(corpus, query="q", favours=1)
        second = encoder_for(corpus, query="q", favours=1)
        for encoder in (first, second):
            encoder.model_name = "fake-model"  # type: ignore[attr-defined]

        RetrievalService(paper, cache=cache, encoder_factory=lambda: first).retriever("dense")
        RetrievalService(paper, cache=cache, encoder_factory=lambda: second).retriever("dense")

        assert first.calls
        assert second.calls == []


class TestDeterminism:
    def test_the_same_query_ranks_identically_across_runs(self, paper: Path) -> None:
        corpus = build_corpus(paper)

        def ranking() -> list[str]:
            factory = factory_for(corpus, encoder_for(corpus, query="adaptive gate", favours=2))
            return [r.chunk_id for r in factory.get("hybrid").retrieve("adaptive gate", top_k=5)]

        assert ranking() == ranking()

    def test_ties_do_not_depend_on_corpus_construction_order(self, paper: Path) -> None:
        # Every chunk embeds to the same vector, so the whole ranking is one big tie.
        corpus = build_corpus(paper)
        flat = FakeEncoder(
            {chunk.text: [1.0, 0.0] for chunk in corpus} | {"q": [1.0, 0.0]}, dimension=2
        )

        reversed_corpus = Corpus(list(corpus)[::-1])

        first = factory_for(corpus, flat).get("dense").retrieve("q", top_k=7)
        second = factory_for(reversed_corpus, flat).get("dense").retrieve("q", top_k=7)

        assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
        assert [r.chunk_id for r in first] == sorted(r.chunk_id for r in first)

    def test_scores_are_finite_everywhere(self, paper: Path) -> None:
        corpus = build_corpus(paper)
        factory = factory_for(corpus, encoder_for(corpus, query="imputed prior", favours=1))

        for name in ("bm25", "dense", "hybrid", "reranked"):
            for result in factory.get(name).retrieve("imputed prior", top_k=5):
                assert np.isfinite(result.score), f"{name} produced {result.score}"
