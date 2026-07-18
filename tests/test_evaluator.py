"""The evaluation runner: aggregation, comparison, and its architectural boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag_eval.chunking import TextChunk
from rag_eval.evaluation import BenchmarkQuestion, EvaluationReport, QueryScore, compare, evaluate
from rag_eval.retrieval import RetrievalResult


class ScriptedRetriever:
    """Returns a fixed ranking per query, so the expected metrics are computable."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings
        self.queries: list[str] = []

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        self.queries.append(query)
        return [
            RetrievalResult(
                chunk=TextChunk(
                    chunk_id=chunk_id,
                    text=chunk_id,
                    source="omics.pdf",
                    page_number=1,
                    chunk_index=0,
                ),
                score=1.0,
                rank=rank,
            )
            for rank, chunk_id in enumerate(self.rankings.get(query, [])[:top_k], start=1)
        ]


def question(
    question_id: str, query: str, relevant: set[str], category: str = "lexical"
) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        question_id=question_id,
        query=query,
        relevant_chunk_ids=frozenset(relevant),
        category=category,
    )


class TestBenchmarkQuestion:
    def test_a_question_with_no_labels_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one relevant chunk"):
            BenchmarkQuestion(question_id="q1", query="anything", relevant_chunk_ids=frozenset())

    def test_an_empty_query_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="query must not be empty"):
            BenchmarkQuestion(question_id="q1", query="  ", relevant_chunk_ids=frozenset({"a"}))

    def test_an_empty_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="question_id must not be empty"):
            BenchmarkQuestion(question_id="", query="q", relevant_chunk_ids=frozenset({"a"}))

    def test_relevant_ids_are_coerced_to_a_frozenset(self) -> None:
        parsed = BenchmarkQuestion(
            question_id="q1", query="q", relevant_chunk_ids=frozenset(["a", "a", "b"])
        )

        assert parsed.relevant_chunk_ids == frozenset({"a", "b"})


class TestEvaluate:
    def test_scores_every_question(self) -> None:
        retriever = ScriptedRetriever({"one": ["a"], "two": ["b"]})
        questions = [question("q1", "one", {"a"}), question("q2", "two", {"b"})]

        report = evaluate(retriever, questions, k=3)

        assert [score.question_id for score in report.scores] == ["q1", "q2"]

    def test_perfect_retrieval_scores_one_on_both_metrics(self) -> None:
        retriever = ScriptedRetriever({"one": ["a", "b"]})

        report = evaluate(retriever, [question("q1", "one", {"a", "b"})], k=2)

        assert report.mean_recall == pytest.approx(1.0)
        assert report.mean_ndcg == pytest.approx(1.0)

    def test_a_total_miss_scores_zero(self) -> None:
        retriever = ScriptedRetriever({"one": ["x", "y"]})

        report = evaluate(retriever, [question("q1", "one", {"a"})], k=2)

        assert report.mean_recall == pytest.approx(0.0)
        assert report.questions_with_no_hit == ("q1",)

    def test_means_are_unweighted_across_questions(self) -> None:
        # q1 has one label and finds it; q2 has four labels and finds one.
        retriever = ScriptedRetriever({"one": ["a"], "two": ["b"]})
        questions = [question("q1", "one", {"a"}), question("q2", "two", {"b", "c", "d", "e"})]

        report = evaluate(retriever, questions, k=4)

        assert report.mean_recall == pytest.approx((1.0 + 0.25) / 2)

    def test_the_retriever_is_queried_once_per_question(self) -> None:
        retriever = ScriptedRetriever({"one": ["a"]})

        evaluate(retriever, [question("q1", "one", {"a"})], k=3)

        assert retriever.queries == ["one"]

    def test_retrieval_uses_the_cutoff_as_top_k(self) -> None:
        # Only "a" fits in the top 1, so the second label cannot be found.
        retriever = ScriptedRetriever({"one": ["a", "b"]})

        report = evaluate(retriever, [question("q1", "one", {"a", "b"})], k=1)

        assert report.scores[0].retrieved_chunk_ids == ("a",)
        assert report.scores[0].recall == pytest.approx(0.5)

    def test_the_report_name_defaults_to_the_class_name(self) -> None:
        report = evaluate(ScriptedRetriever({"one": ["a"]}), [question("q1", "one", {"a"})], k=1)

        assert report.retriever_name == "ScriptedRetriever"

    def test_the_report_name_is_overridable(self) -> None:
        report = evaluate(
            ScriptedRetriever({"one": ["a"]}), [question("q1", "one", {"a"})], k=1, name="bm25"
        )

        assert report.retriever_name == "bm25"

    def test_an_empty_question_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty question set"):
            evaluate(ScriptedRetriever({}), [], k=3)

    def test_a_non_positive_k_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            evaluate(ScriptedRetriever({}), [question("q1", "one", {"a"})], k=0)

    def test_something_that_cannot_retrieve_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="retrieve"):
            evaluate(object(), [question("q1", "one", {"a"})], k=3)  # type: ignore[arg-type]


class TestReportAggregation:
    @pytest.fixture
    def report(self) -> EvaluationReport:
        retriever = ScriptedRetriever({"one": ["a"], "two": ["x"], "three": ["c"]})
        questions = [
            question("q1", "one", {"a"}, category="lexical"),
            question("q2", "two", {"b"}, category="paraphrase"),
            question("q3", "three", {"c"}, category="lexical"),
        ]
        return evaluate(retriever, questions, k=3)

    def test_by_category_groups_the_scores(self, report: EvaluationReport) -> None:
        grouped = report.by_category()

        assert set(grouped) == {"lexical", "paraphrase"}
        assert grouped["lexical"][0] == pytest.approx(1.0)
        assert grouped["paraphrase"][0] == pytest.approx(0.0)

    def test_failures_are_listed(self, report: EvaluationReport) -> None:
        assert report.questions_with_no_hit == ("q2",)

    def test_a_report_needs_at_least_one_score(self) -> None:
        with pytest.raises(ValueError, match="at least one scored question"):
            EvaluationReport(retriever_name="x", k=3, scores=())

    def test_metadata_is_read_only(self) -> None:
        report = EvaluationReport(
            retriever_name="x",
            k=1,
            scores=(QueryScore("q1", "lexical", 1, 1.0, 1.0),),
            metadata={"chunker": "fixed"},
        )

        with pytest.raises(TypeError):
            report.metadata["chunker"] = "structure"  # type: ignore[index]


class TestCompare:
    def test_evaluates_every_retriever_on_the_same_questions(self) -> None:
        questions = [question("q1", "one", {"a"})]
        reports = compare(
            {
                "good": ScriptedRetriever({"one": ["a"]}),
                "bad": ScriptedRetriever({"one": ["x"]}),
            },
            questions,
            k=3,
        )

        assert [report.retriever_name for report in reports] == ["good", "bad"]
        assert reports[0].mean_recall > reports[1].mean_recall

    def test_reports_keep_insertion_order_rather_than_ranking_by_score(self) -> None:
        # Sorting by score would invite reading row one as "the winner".
        reports = compare(
            {
                "bad": ScriptedRetriever({"one": ["x"]}),
                "good": ScriptedRetriever({"one": ["a"]}),
            },
            [question("q1", "one", {"a"})],
            k=3,
        )

        assert [report.retriever_name for report in reports] == ["bad", "good"]

    def test_an_empty_retriever_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty set of retrievers"):
            compare({}, [question("q1", "one", {"a"})], k=3)


class TestArchitecturalBoundaries:
    def test_evaluation_does_not_import_generation_or_streamlit(self) -> None:
        # Rule 1 and rule 2 of the project, enforced rather than trusted.
        import rag_eval

        source_root = Path(rag_eval.__file__).resolve().parents[1]
        code = (
            "import sys; import rag_eval.evaluation; "
            "print(any(name.startswith('rag_eval.generation') or name == 'streamlit' "
            "for name in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(source_root)},
        )

        assert completed.stdout.strip() == "False"
