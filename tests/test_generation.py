"""Grounded generation: citations that resolve, refusals that fire, and no re-retrieval."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag_eval.chunking import TextChunk
from rag_eval.cli import main
from rag_eval.generation import (
    INSUFFICIENT_EVIDENCE,
    NO_EVIDENCE_TEXT,
    SYSTEM_PROMPT,
    AnswerGenerator,
    ClaudeLanguageModel,
    Evidence,
    GenerationError,
    LanguageModel,
    build_prompt,
    evidence_from_results,
)
from rag_eval.generation import llm as llm_module
from rag_eval.retrieval import RetrievalResult
from tests.fakes import FakeLanguageModel


def result(marker: int, *, text: str | None = None, section: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk=TextChunk(
            chunk_id=f"omics-p{marker:03d}-c00-abcdef12",
            text=text or f"passage {marker}",
            source="omics.pdf",
            page_number=marker,
            chunk_index=0,
            section=section,
        ),
        score=1.0 / marker,
        rank=marker,
    )


def results(count: int) -> list[RetrievalResult]:
    return [result(index) for index in range(1, count + 1)]


class TestEvidence:
    def test_results_are_numbered_from_one(self) -> None:
        evidence = evidence_from_results(results(3))

        assert [item.marker for item in evidence] == [1, 2, 3]

    def test_retrieval_order_is_preserved(self) -> None:
        # The retriever already decided what is most relevant; generation consumes
        # that decision rather than second-guessing it.
        evidence = evidence_from_results(results(3))

        assert [item.chunk_id for item in evidence] == [item.chunk_id for item in results(3)]

    def test_markers_are_one_indexed(self) -> None:
        with pytest.raises(ValueError, match="1-indexed"):
            Evidence(marker=0, result=result(1))

    def test_no_results_means_no_evidence(self) -> None:
        assert evidence_from_results([]) == ()

    def test_provenance_survives_into_the_citation(self) -> None:
        citation = Evidence(marker=1, result=result(3, section="Methods")).to_citation()

        assert citation.chunk_id == "omics-p003-c00-abcdef12"
        assert citation.page_number == 3
        assert citation.label == "omics.pdf p.3 § Methods"

    def test_a_citation_without_a_section_still_names_its_page(self) -> None:
        assert Evidence(marker=1, result=result(2)).to_citation().label == "omics.pdf p.2"


class TestPrompt:
    def test_every_passage_is_numbered_in_the_prompt(self) -> None:
        prompt = build_prompt("how?", evidence_from_results(results(3)))

        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_passages_carry_their_provenance(self) -> None:
        prompt = build_prompt("how?", evidence_from_results([result(4, section="Results")]))

        assert "omics.pdf p.4 § Results" in prompt

    def test_retrieval_scores_are_not_shown_to_the_model(self) -> None:
        # A score is a within-retriever artifact that means nothing to a model, and
        # showing it invites treating rank as truth.
        prompt = build_prompt("how?", evidence_from_results(results(2)))

        assert "0.5" not in prompt
        assert "score" not in prompt.lower()

    def test_the_question_is_included(self) -> None:
        assert "how are modalities imputed?" in build_prompt(
            "how are modalities imputed?", evidence_from_results(results(1))
        )

    def test_a_prompt_with_no_evidence_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no evidence"):
            build_prompt("how?", [])

    def test_the_system_prompt_names_the_refusal_sentinel(self) -> None:
        assert INSUFFICIENT_EVIDENCE in SYSTEM_PROMPT


class TestAnswering:
    def test_an_answer_resolves_its_citations(self) -> None:
        generator = AnswerGenerator(FakeLanguageModel("Imputed with a learned prior [2]."))

        answer = generator.answer("how?", results(3))

        assert [citation.marker for citation in answer.citations] == [2]
        assert answer.citations[0].page_number == 2
        assert answer.is_grounded is True

    def test_citations_appear_in_first_use_order_and_deduplicate(self) -> None:
        generator = AnswerGenerator(FakeLanguageModel("A [3], b [1], c [3] again."))

        answer = generator.answer("how?", results(3))

        assert [citation.marker for citation in answer.citations] == [3, 1]

    @pytest.mark.parametrize("reply", ["Both [1][3].", "Both [1, 3].", "Both [1; 3]."])
    def test_the_shapes_models_actually_emit_all_parse(self, reply: str) -> None:
        answer = AnswerGenerator(FakeLanguageModel(reply)).answer("how?", results(3))

        assert [citation.marker for citation in answer.citations] == [1, 3]

    def test_cited_pages_are_deduplicated(self) -> None:
        answer = AnswerGenerator(FakeLanguageModel("A [1], b [2], c [1].")).answer(
            "how?", results(2)
        )

        assert answer.cited_pages == (("omics.pdf", 1), ("omics.pdf", 2))

    def test_the_formatted_answer_lists_its_sources(self) -> None:
        answer = AnswerGenerator(FakeLanguageModel("A learned prior [2].")).answer(
            "how?", results(3)
        )

        assert "sources:" in answer.format()
        assert "[2] omics.pdf p.2" in answer.format()

    def test_only_the_requested_number_of_passages_is_grounded_on(self) -> None:
        generator = AnswerGenerator(FakeLanguageModel(), max_evidence=2)

        answer = generator.answer("how?", results(5))

        assert len(answer.evidence) == 2

    def test_the_evidence_given_is_reported_back(self) -> None:
        answer = AnswerGenerator(FakeLanguageModel()).answer("how?", results(3))

        assert [item.chunk_id for item in answer.evidence] == [
            item.chunk_id for item in results(3)
        ]


class TestHallucinatedCitations:
    def test_a_marker_with_no_evidence_is_reported_not_dropped(self) -> None:
        # The failure mode the whole pipeline exists to make visible.
        answer = AnswerGenerator(FakeLanguageModel("Claimed [7].")).answer("how?", results(3))

        assert answer.unresolved_citations == (7,)
        assert answer.citations == ()

    def test_an_answer_with_an_unresolved_citation_is_not_grounded(self) -> None:
        answer = AnswerGenerator(FakeLanguageModel("Real [1], invented [9].")).answer(
            "how?", results(3)
        )

        assert answer.citations[0].marker == 1
        assert answer.unresolved_citations == (9,)
        assert answer.is_grounded is False

    def test_an_uncited_answer_is_not_grounded(self) -> None:
        answer = AnswerGenerator(FakeLanguageModel("It works, trust me.")).answer(
            "how?", results(3)
        )

        assert answer.citations == ()
        assert answer.is_grounded is False
        assert answer.has_answer is True


class TestInsufficientEvidence:
    def test_no_results_means_no_model_call_at_all(self) -> None:
        # Nothing to ground an answer in, so asking would only buy an unsourced one.
        model = FakeLanguageModel()

        answer = AnswerGenerator(model).answer("how?", [])

        assert model.calls == []
        assert answer.has_answer is False
        assert answer.text == NO_EVIDENCE_TEXT

    def test_the_refusal_sentinel_becomes_a_refusal(self) -> None:
        answer = AnswerGenerator(FakeLanguageModel(INSUFFICIENT_EVIDENCE)).answer(
            "how?", results(3)
        )

        assert answer.has_answer is False
        assert answer.is_grounded is False
        assert answer.text == NO_EVIDENCE_TEXT

    def test_a_refusal_still_reports_the_evidence_it_saw(self) -> None:
        answer = AnswerGenerator(FakeLanguageModel(INSUFFICIENT_EVIDENCE)).answer(
            "how?", results(3)
        )

        assert len(answer.evidence) == 3

    def test_an_answer_merely_containing_the_sentinel_is_not_a_refusal(self) -> None:
        reply = f"The passages say {INSUFFICIENT_EVIDENCE} was the label used [1]."

        answer = AnswerGenerator(FakeLanguageModel(reply)).answer("how?", results(2))

        assert answer.has_answer is True

    def test_a_refusal_formats_as_plain_text(self) -> None:
        answer = AnswerGenerator(FakeLanguageModel(INSUFFICIENT_EVIDENCE)).answer(
            "how?", results(2)
        )

        assert answer.format() == NO_EVIDENCE_TEXT


class TestValidation:
    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_queries_are_rejected(self, query: str) -> None:
        with pytest.raises(ValueError, match="query must not be empty"):
            AnswerGenerator(FakeLanguageModel()).answer(query, results(2))

    def test_max_evidence_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_evidence must be positive"):
            AnswerGenerator(FakeLanguageModel(), max_evidence=0)

    @pytest.mark.parametrize("reply", ["", "   \n  "])
    def test_an_empty_completion_is_an_error_not_a_refusal(self, reply: str) -> None:
        # A model that says nothing is broken; a model that refuses says so.
        with pytest.raises(GenerationError, match="empty answer"):
            AnswerGenerator(FakeLanguageModel(reply)).answer("how?", results(2))

    def test_the_fake_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeLanguageModel(), LanguageModel)


class TestClaudeClient:
    def test_a_missing_key_fails_at_construction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Better than failing after a document has already been indexed.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with pytest.raises(GenerationError, match="no API key"):
            ClaudeLanguageModel()

    def test_a_blank_key_is_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")

        with pytest.raises(GenerationError, match="no API key"):
            ClaudeLanguageModel()

    def test_max_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ClaudeLanguageModel(api_key="test-key", max_tokens=0)

    def test_text_blocks_are_concatenated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        body = {
            "content": [
                {"type": "text", "text": "A learned prior "},
                {"type": "text", "text": "[1]."},
            ]
        }
        self._stub_response(monkeypatch, body)

        assert ClaudeLanguageModel(api_key="test-key").complete("sys", "hi") == (
            "A learned prior [1]."
        )

    def test_the_request_carries_the_key_and_the_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        self._stub_response(
            monkeypatch, {"content": [{"type": "text", "text": "ok"}]}, captured=captured
        )

        ClaudeLanguageModel(api_key="secret", model="test-model").complete("sys", "hi")

        request = captured["request"]
        payload = json.loads(captured["body"])  # type: ignore[arg-type]
        assert request.headers["X-api-key"] == "secret"  # type: ignore[union-attr]
        assert payload["system"] == "sys"
        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.0

    def test_a_response_with_no_text_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_response(monkeypatch, {"content": [{"type": "thinking"}]})

        with pytest.raises(GenerationError, match="empty completion"):
            ClaudeLanguageModel(api_key="test-key").complete("sys", "hi")

    def test_a_malformed_response_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._stub_response(monkeypatch, {"error": "nope"})

        with pytest.raises(GenerationError, match="no content blocks"):
            ClaudeLanguageModel(api_key="test-key").complete("sys", "hi")

    @staticmethod
    def _stub_response(
        monkeypatch: pytest.MonkeyPatch,
        body: dict[str, object],
        *,
        captured: dict[str, object] | None = None,
    ) -> None:
        """Replace ``urlopen`` so the client is exercised without a network."""

        class Response(io.BytesIO):
            def __enter__(self) -> Response:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def fake_urlopen(request: object, timeout: float = 0.0) -> Response:
            if captured is not None:
                captured["request"] = request
                captured["body"] = request.data.decode("utf-8")  # type: ignore[union-attr]
            return Response(json.dumps(body).encode("utf-8"))

        monkeypatch.setattr(llm_module.urllib.request, "urlopen", fake_urlopen)


class TestArchitecturalBoundaries:
    def test_evaluation_does_not_import_generation(self) -> None:
        # Retrieval quality is measured on its own terms. If evaluation could reach
        # generation, a benchmark could quietly start scoring answers instead of ranks.
        assert self._imports_are_absent(
            "import rag_eval.evaluation", "rag_eval.generation"
        )

    def test_retrieval_does_not_import_generation(self) -> None:
        assert self._imports_are_absent("import rag_eval.retrieval", "rag_eval.generation")

    def test_importing_generation_costs_no_model_libraries(self) -> None:
        assert self._imports_are_absent("import rag_eval.generation", "torch")

    @staticmethod
    def _imports_are_absent(setup: str, forbidden: str) -> bool:
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


class TestAskCommand:
    PAGES = [
        "Introduction\nThe multi-omics embedding is disease aware and adaptive.",
        "Methods\nMissing modalities are imputed with a learned prior across cohorts.",
        "Results\nThe adaptive gate improved downstream survival prediction accuracy.",
        "Discussion\nAblations show the disease aware routing contributes most of the gain.",
        "Related Work\nEarlier fusion approaches concatenate modality features naively.",
        "Conclusion\nAn agentic framework for adaptive multi-omics embedding is presented.",
        "Appendix\nHyperparameters were selected by grid search over the validation split.",
    ]

    @pytest.fixture
    def paper(self, make_pdf: object) -> Path:
        return make_pdf(self.PAGES, name="paper.pdf")  # type: ignore[operator]

    def test_a_missing_key_exits_nonzero_with_a_clear_error(
        self, paper: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        exit_code = main(["ask", str(paper), "imputation prior", "-r", "bm25"])

        assert exit_code == 1
        assert "no API key" in capsys.readouterr().err

    def test_an_answer_is_printed_with_its_sources(
        self, paper: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        TestClaudeClient._stub_response(
            monkeypatch, {"content": [{"type": "text", "text": "A learned prior [1]."}]}
        )

        exit_code = main(["ask", str(paper), "imputed prior", "-r", "bm25"])

        assert exit_code == 0
        assert "sources:" in capsys.readouterr().out

    def test_a_hallucinated_citation_exits_nonzero(
        self, paper: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        TestClaudeClient._stub_response(
            monkeypatch, {"content": [{"type": "text", "text": "Claimed [9]."}]}
        )

        exit_code = main(["ask", str(paper), "imputed prior", "-r", "bm25"])

        assert exit_code == 1
        assert "never supplied as evidence" in capsys.readouterr().out

    def test_evidence_can_be_shown_alongside_the_answer(
        self, paper: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        TestClaudeClient._stub_response(
            monkeypatch, {"content": [{"type": "text", "text": "A learned prior [1]."}]}
        )

        main(["ask", str(paper), "imputed prior", "-r", "bm25", "--show-evidence"])

        assert "evidence (" in capsys.readouterr().out
