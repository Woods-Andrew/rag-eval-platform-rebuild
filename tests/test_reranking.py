"""Cross-encoder reranking: reordering, candidate depth, and what it cannot do."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from rag_eval.chunking import TextChunk
from rag_eval.evaluation import BenchmarkQuestion, evaluate
from rag_eval.reranking import Reranker, RerankingRetriever
from rag_eval.retrieval import RetrievalResult, Retriever
from tests.fakes import FakeReranker


def chunk(chunk_id: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id, text=chunk_id, source="omics.pdf", page_number=1, chunk_index=0
    )


class ScriptedRetriever:
    """Returns a fixed ranking, recording the depth it was asked for."""

    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids
        self.requested_top_k: list[int] = []

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        self.requested_top_k.append(top_k)
        return [
            RetrievalResult(chunk=chunk(chunk_id), score=1.0 / rank, rank=rank)
            for rank, chunk_id in enumerate(self.chunk_ids[:top_k], start=1)
        ]


def reranker_for(query: str, ordering: list[str]) -> FakeReranker:
    """A reranker that prefers ``ordering``, best first."""
    return FakeReranker(
        {(query, chunk_id): float(len(ordering) - index) for index, chunk_id in enumerate(ordering)}
    )


class TestReordering:
    def test_candidates_are_reordered_by_the_reranker(self) -> None:
        base = ScriptedRetriever(["a", "b", "c"])
        reranked = RerankingRetriever(base, reranker_for("q", ["c", "a", "b"]))

        results = reranked.retrieve("q", top_k=3)

        assert [result.chunk_id for result in results] == ["c", "a", "b"]

    def test_ranks_are_renumbered_from_one(self) -> None:
        reranked = RerankingRetriever(
            ScriptedRetriever(["a", "b", "c"]), reranker_for("q", ["c", "b", "a"])
        )

        assert [result.rank for result in reranked.retrieve("q", top_k=3)] == [1, 2, 3]

    def test_the_score_becomes_the_reranker_score(self) -> None:
        reranked = RerankingRetriever(
            ScriptedRetriever(["a", "b"]), reranker_for("q", ["b", "a"])
        )

        results = reranked.retrieve("q", top_k=2)

        assert results[0].score == pytest.approx(2.0)
        assert results[1].score == pytest.approx(1.0)

    def test_output_is_truncated_to_top_k(self) -> None:
        reranked = RerankingRetriever(
            ScriptedRetriever(["a", "b", "c", "d"]), reranker_for("q", ["d", "c", "b", "a"])
        )

        assert [r.chunk_id for r in reranked.retrieve("q", top_k=2)] == ["d", "c"]

    def test_ties_break_on_chunk_id_not_candidate_order(self) -> None:
        forwards = RerankingRetriever(ScriptedRetriever(["b", "a"]), FakeReranker({}))
        backwards = RerankingRetriever(ScriptedRetriever(["a", "b"]), FakeReranker({}))

        assert [r.chunk_id for r in forwards.retrieve("q", top_k=2)] == ["a", "b"]
        assert [r.chunk_id for r in backwards.retrieve("q", top_k=2)] == ["a", "b"]

    def test_provenance_survives_reranking(self) -> None:
        reranked = RerankingRetriever(ScriptedRetriever(["a"]), reranker_for("q", ["a"]))

        assert reranked.retrieve("q", top_k=1)[0].citation == "omics.pdf p.1"


class TestCandidateDepth:
    def test_the_reranker_is_given_more_candidates_than_requested(self) -> None:
        base = ScriptedRetriever(["a"])

        RerankingRetriever(base, FakeReranker({}), candidate_multiplier=5).retrieve("q", top_k=4)

        assert base.requested_top_k == [20]

    def test_the_multiplier_is_configurable(self) -> None:
        base = ScriptedRetriever(["a"])

        RerankingRetriever(base, FakeReranker({}), candidate_multiplier=1).retrieve("q", top_k=4)

        assert base.requested_top_k == [4]

    def test_every_candidate_is_scored_in_one_call(self) -> None:
        reranker = FakeReranker({})
        base = ScriptedRetriever(["a", "b", "c"])

        RerankingRetriever(base, reranker, candidate_multiplier=3).retrieve("q", top_k=1)

        assert len(reranker.calls) == 1
        assert reranker.calls[0][1] == ("a", "b", "c")

    def test_a_chunk_outside_the_candidate_window_can_never_be_promoted(self) -> None:
        # The structural limit of the two-stage design: reranking reorders, it does
        # not retrieve. "z" is rank 3 and the window is 2, so no reranker can save it.
        base = ScriptedRetriever(["a", "b", "z"])
        reranked = RerankingRetriever(
            base, reranker_for("q", ["z", "a", "b"]), candidate_multiplier=2
        )

        assert [r.chunk_id for r in reranked.retrieve("q", top_k=1)] == ["a"]


class TestEffectOnMetrics:
    def test_reranking_lifts_ndcg_while_recall_stays_flat(self) -> None:
        # The reranker's whole job, measured. Both retrievers surface the relevant
        # chunk within K, so recall is identical; only the ordering differs.
        base = ScriptedRetriever(["x", "y", "a"])
        reranked = RerankingRetriever(
            base, reranker_for("q", ["a", "x", "y"]), candidate_multiplier=1
        )
        questions = [
            BenchmarkQuestion(
                question_id="q1", query="q", relevant_chunk_ids=frozenset({"a"})
            )
        ]

        before = evaluate(ScriptedRetriever(["x", "y", "a"]), questions, k=3)
        after = evaluate(reranked, questions, k=3)

        assert after.mean_recall == pytest.approx(before.mean_recall)
        assert after.mean_ndcg > before.mean_ndcg

    def test_reranking_cannot_raise_recall_above_the_base_retriever(self) -> None:
        base = ScriptedRetriever(["x", "y"])  # never returns the relevant chunk
        reranked = RerankingRetriever(base, reranker_for("q", ["x", "y"]))
        questions = [
            BenchmarkQuestion(
                question_id="q1", query="q", relevant_chunk_ids=frozenset({"a"})
            )
        ]

        assert evaluate(reranked, questions, k=2).mean_recall == pytest.approx(0.0)


class TestComposition:
    def test_satisfies_the_retriever_protocol(self) -> None:
        reranked = RerankingRetriever(ScriptedRetriever(["a"]), FakeReranker({}))

        assert isinstance(reranked, Retriever)

    def test_the_fake_satisfies_the_reranker_protocol(self) -> None:
        assert isinstance(FakeReranker({}), Reranker)

    def test_the_base_retriever_is_exposed(self) -> None:
        base = ScriptedRetriever(["a"])

        assert RerankingRetriever(base, FakeReranker({})).base_retriever is base

    def test_a_reranker_can_wrap_a_reranker(self) -> None:
        inner = RerankingRetriever(ScriptedRetriever(["a", "b"]), reranker_for("q", ["b", "a"]))
        outer = RerankingRetriever(inner, reranker_for("q", ["a", "b"]))

        assert [r.chunk_id for r in outer.retrieve("q", top_k=2)] == ["a", "b"]


class TestEdgeCases:
    def test_no_candidates_means_no_results_and_no_reranker_call(self) -> None:
        reranker = FakeReranker({})

        assert RerankingRetriever(ScriptedRetriever([]), reranker).retrieve("q", top_k=3) == []
        assert reranker.calls == []

    def test_a_score_count_mismatch_is_rejected(self) -> None:
        class ShortReranker:
            def score(self, query: str, passages: list[str]) -> list[float]:
                return [1.0]

        reranked = RerankingRetriever(ScriptedRetriever(["a", "b"]), ShortReranker())

        with pytest.raises(ValueError, match="1 scores for 2 passages"):
            reranked.retrieve("q", top_k=2)

    def test_candidate_multiplier_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="candidate_multiplier must be at least 1"):
            RerankingRetriever(ScriptedRetriever(["a"]), FakeReranker({}), candidate_multiplier=0)

    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_queries_are_rejected(self, query: str) -> None:
        reranked = RerankingRetriever(ScriptedRetriever(["a"]), FakeReranker({}))

        with pytest.raises(ValueError, match="query must not be empty"):
            reranked.retrieve(query, top_k=3)

    def test_non_positive_top_k_is_rejected(self) -> None:
        reranked = RerankingRetriever(ScriptedRetriever(["a"]), FakeReranker({}))

        with pytest.raises(ValueError, match="top_k must be positive"):
            reranked.retrieve("q", top_k=0)


class TestOfflineBoundary:
    def test_importing_reranking_does_not_import_the_model_library(self) -> None:
        import os

        import rag_eval

        source_root = Path(rag_eval.__file__).resolve().parents[1]
        code = (
            "import sys; import rag_eval.reranking; "
            "print('sentence_transformers' in sys.modules or 'torch' in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(source_root)},
        )

        assert completed.stdout.strip() == "False"
