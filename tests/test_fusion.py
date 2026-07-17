"""Reciprocal rank fusion: the arithmetic, verified by hand."""

from __future__ import annotations

import pytest

from rag_eval.chunking import TextChunk
from rag_eval.retrieval import (
    DEFAULT_RRF_K,
    RetrievalResult,
    reciprocal_rank_fusion,
    rrf_scores,
)


def ranked(*chunk_ids: str, scores: list[float] | None = None) -> list[RetrievalResult]:
    """A ranked list from chunk IDs, ranks assigned 1..n in the order given."""
    return [
        RetrievalResult(
            chunk=TextChunk(
                chunk_id=chunk_id,
                text=chunk_id,
                source="omics.pdf",
                page_number=1,
                chunk_index=0,
            ),
            score=scores[index] if scores else 1.0,
            rank=index + 1,
        )
        for index, chunk_id in enumerate(chunk_ids)
    ]


class TestRRFArithmetic:
    def test_a_single_list_scores_one_over_k_plus_rank(self) -> None:
        scores = rrf_scores([ranked("a", "b")], k=60)

        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 62)

    def test_contributions_from_both_lists_are_summed(self) -> None:
        scores = rrf_scores([ranked("a", "b"), ranked("b", "a")], k=60)

        assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
        assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)

    def test_a_chunk_missing_from_one_list_contributes_only_the_other_term(self) -> None:
        scores = rrf_scores([ranked("a"), ranked("b")], k=60)

        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["b"] == pytest.approx(1 / 61)

    def test_the_default_k_is_sixty(self) -> None:
        assert DEFAULT_RRF_K == 60
        assert rrf_scores([ranked("a")]) == pytest.approx({"a": 1 / 61})

    def test_smaller_k_sharpens_the_advantage_of_rank_one(self) -> None:
        sharp = rrf_scores([ranked("a", "b")], k=1)
        damped = rrf_scores([ranked("a", "b")], k=60)

        assert sharp["a"] / sharp["b"] > damped["a"] / damped["b"]

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            rrf_scores([ranked("a")], k=0)


class TestFusionBehaviour:
    def test_agreement_beats_a_single_top_rank(self) -> None:
        # "b" is nobody's favourite but is second on both lists; "a" and "c" are each
        # first on one list and absent from the other. Damped k lets agreement win.
        lexical = ranked("a", "b")
        dense = ranked("c", "b")

        fused = reciprocal_rank_fusion([lexical, dense], top_k=3)

        assert fused[0].chunk_id == "b"

    def test_a_chunk_found_by_only_one_retriever_still_appears(self) -> None:
        fused = reciprocal_rank_fusion([ranked("a"), ranked("b")], top_k=2)

        assert {result.chunk_id for result in fused} == {"a", "b"}

    def test_raw_scores_are_ignored_entirely(self) -> None:
        # Same ranks, wildly different scores — BM25-scale versus cosine-scale.
        huge = ranked("a", "b", scores=[912.4, 400.1])
        small = ranked("a", "b", scores=[0.81, 0.42])

        assert rrf_scores([huge]) == pytest.approx(rrf_scores([small]))

    def test_fused_ranks_are_renumbered_from_one(self) -> None:
        fused = reciprocal_rank_fusion([ranked("a", "b", "c")], top_k=3)

        assert [result.rank for result in fused] == [1, 2, 3]

    def test_fused_scores_are_non_increasing(self) -> None:
        fused = reciprocal_rank_fusion([ranked("a", "b"), ranked("b", "c")], top_k=3)

        scores = [result.score for result in fused]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates_the_fused_list(self) -> None:
        fused = reciprocal_rank_fusion([ranked("a", "b", "c")], top_k=2)

        assert [result.chunk_id for result in fused] == ["a", "b"]

    def test_the_order_of_the_input_lists_does_not_matter(self) -> None:
        forwards = reciprocal_rank_fusion([ranked("a", "b"), ranked("b", "a")], top_k=2)
        backwards = reciprocal_rank_fusion([ranked("b", "a"), ranked("a", "b")], top_k=2)

        assert [r.chunk_id for r in forwards] == [r.chunk_id for r in backwards]

    def test_the_fused_result_carries_the_original_chunk(self) -> None:
        fused = reciprocal_rank_fusion([ranked("a")], top_k=1)

        assert fused[0].chunk.chunk_id == "a"
        assert fused[0].citation == "omics.pdf p.1"

    def test_empty_lists_fuse_to_nothing(self) -> None:
        assert reciprocal_rank_fusion([[], []], top_k=5) == []

    def test_top_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="top_k must be positive"):
            reciprocal_rank_fusion([ranked("a")], top_k=0)
