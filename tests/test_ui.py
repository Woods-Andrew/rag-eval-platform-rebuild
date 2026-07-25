"""The UI service layer, the Streamlit app, and the boundary between them."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag_eval.chunking import FixedSizeChunker, StructureAwareChunker
from rag_eval.generation import INSUFFICIENT_EVIDENCE
from rag_eval.retrieval import TextEncoder
from rag_eval.ui import RETRIEVERS, RetrievalService, make_chunker
from tests.fakes import FakeEncoder, FakeLanguageModel, FakeReranker

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

APP_SCRIPT = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def paper(make_pdf: PdfFactory) -> Path:
    return make_pdf(PAGES, name="paper.pdf")


class CountingEncoderFactory:
    """Hands out one fake encoder and counts how often it was asked for one."""

    def __init__(self) -> None:
        self.calls = 0
        self.encoder = FakeEncoder({}, dimension=4)

    def __call__(self) -> TextEncoder:
        self.calls += 1
        return self.encoder


def service_for(
    paper: Path, *, model: FakeLanguageModel | None = None, **kwargs: object
) -> RetrievalService:
    return RetrievalService(
        paper,
        encoder_factory=CountingEncoderFactory(),
        reranker_factory=lambda: FakeReranker({}),
        model_factory=(lambda: model) if model else None,
        **kwargs,  # type: ignore[arg-type]
    )


class TestChunkerSelection:
    def test_known_names_build_their_chunkers(self) -> None:
        assert isinstance(make_chunker("fixed"), FixedSizeChunker)
        assert isinstance(make_chunker("structure"), StructureAwareChunker)

    def test_an_unknown_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown chunker"):
            make_chunker("semantic")


class TestIndexing:
    def test_the_document_is_indexed_once_at_construction(self, paper: Path) -> None:
        service = service_for(paper)

        assert len(service.corpus) == len(PAGES)
        assert service.document == "paper.pdf"

    def test_corpus_statistics_are_available_without_a_query(self, paper: Path) -> None:
        # The sidebar shows these before anything has been searched.
        assert service_for(paper).stats.chunk_count == len(PAGES)

    def test_the_chunker_is_selectable(self, paper: Path) -> None:
        structure = RetrievalService(
            paper, StructureAwareChunker(min_words=1), encoder_factory=CountingEncoderFactory()
        )

        assert any(chunk.section for chunk in structure.corpus)


class TestRetrieverCaching:
    def test_nothing_is_built_until_a_strategy_is_selected(self, paper: Path) -> None:
        service = service_for(paper)

        assert service.built_retrievers == ()

    def test_a_retriever_is_built_once_and_reused(self, paper: Path) -> None:
        service = service_for(paper)

        first = service.retriever("bm25")
        second = service.retriever("bm25")

        assert first is second

    def test_switching_strategies_never_re_encodes_the_corpus(self, paper: Path) -> None:
        # The rule the UI is most likely to break: a radio button must not trigger a
        # second embedding pass over the whole document.
        factory = CountingEncoderFactory()
        service = RetrievalService(
            paper,
            encoder_factory=factory,
            reranker_factory=lambda: FakeReranker({}),
        )

        service.retriever("dense")
        service.retriever("hybrid")
        service.retriever("reranked")
        service.retriever("dense")

        assert factory.calls == 1

    def test_hybrid_reuses_the_cached_component_retrievers(self, paper: Path) -> None:
        service = service_for(paper)

        service.retriever("hybrid")

        assert set(service.built_retrievers) == {"bm25", "dense", "hybrid"}

    def test_reranked_reuses_the_cached_hybrid(self, paper: Path) -> None:
        service = service_for(paper)

        service.retriever("reranked")

        assert set(service.built_retrievers) == {"bm25", "dense", "hybrid", "reranked"}

    @pytest.mark.parametrize("name", RETRIEVERS)
    def test_every_offered_strategy_can_be_built(self, paper: Path, name: str) -> None:
        assert service_for(paper).retriever(name) is not None

    def test_an_unknown_strategy_is_rejected(self, paper: Path) -> None:
        with pytest.raises(ValueError, match="unknown retriever"):
            service_for(paper).retriever("magic")


class TestSearching:
    def test_a_query_returns_ranked_results(self, paper: Path) -> None:
        results = service_for(paper).search("imputed prior", retriever="bm25", top_k=3)

        assert results
        assert [result.rank for result in results] == list(range(1, len(results) + 1))

    def test_validation_is_not_duplicated_in_the_service(self, paper: Path) -> None:
        # The retriever already rejects these; the service must not silently allow them.
        with pytest.raises(ValueError, match="query must not be empty"):
            service_for(paper).search("  ", retriever="bm25", top_k=3)

    def test_top_k_is_rejected_when_non_positive(self, paper: Path) -> None:
        with pytest.raises(ValueError, match="top_k must be positive"):
            service_for(paper).search("prior", retriever="bm25", top_k=0)


class TestAnswering:
    def test_retrieval_alone_needs_no_language_model(self, paper: Path) -> None:
        outcome = service_for(paper).ask("imputed prior", retriever="bm25", top_k=3)

        assert outcome.found_anything is True
        assert outcome.answer is None

    def test_generation_is_skipped_when_not_requested(self, paper: Path) -> None:
        model = FakeLanguageModel()

        outcome = service_for(paper, model=model).ask(
            "imputed prior", retriever="bm25", top_k=3, generate=False
        )

        assert model.calls == []
        assert outcome.answer is None

    def test_an_answer_is_built_from_exactly_what_was_retrieved(self, paper: Path) -> None:
        # No second, hidden retrieval behind the generated text.
        model = FakeLanguageModel("A learned prior [1].")

        outcome = service_for(paper, model=model).ask(
            "imputed prior", retriever="bm25", top_k=3
        )

        assert outcome.answer is not None
        assert [item.chunk_id for item in outcome.answer.evidence] == [
            result.chunk_id for result in outcome.results
        ]

    def test_a_refusal_survives_into_the_outcome(self, paper: Path) -> None:
        model = FakeLanguageModel(INSUFFICIENT_EVIDENCE)

        outcome = service_for(paper, model=model).ask(
            "imputed prior", retriever="bm25", top_k=3
        )

        assert outcome.answer is not None
        assert outcome.answer.has_answer is False

    def test_a_query_matching_nothing_returns_no_results(self, paper: Path) -> None:
        outcome = service_for(paper).ask(
            "zzzznonexistentterm", retriever="bm25", top_k=3
        )

        assert outcome.found_anything is False


class TestStreamlitApp:
    def test_the_app_renders_an_empty_state_without_documents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The state a fresh clone starts in: source PDFs are gitignored, so the app
        # has to say so rather than raise.
        from rag_eval.ui import app as app_module

        monkeypatch.setattr(app_module, "DOCUMENTS", tmp_path / "documents")
        app = AppTest.from_file(APP_SCRIPT)
        app.run(timeout=30)

        assert not app.exception
        assert "No PDFs found" in app.info[0].value

    def test_the_app_indexes_a_document_and_shows_its_shape(
        self, paper: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = self._app_with_documents(paper, monkeypatch)
        app.run(timeout=60)

        assert not app.exception
        assert app.title[0].value == "RAG Evaluation Platform"
        assert app.metric[0].value == str(len(PAGES))

    def test_searching_through_the_app_renders_passages(
        self, paper: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = self._app_with_documents(paper, monkeypatch)
        app.run(timeout=60)

        # bm25 keeps this offline: no embedding model is ever constructed.
        app.sidebar.radio[1].set_value("bm25")
        app.text_input[0].set_value("imputed prior")
        app.run(timeout=60)

        assert not app.exception
        assert app.subheader[0].value.startswith("Retrieved passages")
        assert app.expander

    def test_generation_is_refused_without_a_key(
        self, paper: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        app = self._app_with_documents(paper, monkeypatch)
        app.run(timeout=60)

        app.sidebar.radio[1].set_value("bm25")
        app.sidebar.checkbox[0].set_value(True)
        app.text_input[0].set_value("imputed prior")
        app.run(timeout=60)

        assert not app.exception
        assert "ANTHROPIC_API_KEY" in app.warning[0].value

    @staticmethod
    def _app_with_documents(paper: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
        """Point the app at the fixture PDF and start from a cold cache."""
        import streamlit as st

        from rag_eval.ui import app as app_module

        st.cache_resource.clear()
        monkeypatch.setattr(app_module, "DOCUMENTS", paper.parent)
        return AppTest.from_file(APP_SCRIPT)


class TestArchitecturalBoundaries:
    def test_retrieval_does_not_import_streamlit(self) -> None:
        assert self._absent("import rag_eval.retrieval", "streamlit")

    def test_evaluation_does_not_import_streamlit(self) -> None:
        assert self._absent("import rag_eval.evaluation", "streamlit")

    def test_the_cli_does_not_import_streamlit(self) -> None:
        assert self._absent("import rag_eval.cli", "streamlit")

    def test_the_service_layer_does_not_import_streamlit(self) -> None:
        # The UI is a consumer of retrieval, never a dependency of it — and the
        # service layer is where that separation is easiest to lose by accident.
        assert self._absent("import rag_eval.ui", "streamlit")

    @staticmethod
    def _absent(setup: str, forbidden: str) -> bool:
        import rag_eval

        source_root = Path(rag_eval.__file__).resolve().parents[1]
        code = f"import sys; {setup}; print({forbidden!r} in sys.modules)"
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(source_root)},
        )
        return completed.stdout.strip() == "False"
