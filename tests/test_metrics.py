"""Recall@K and nDCG@K, checked against values computed by hand."""

from __future__ import annotations

import math

import pytest

from rag_eval.evaluation import dcg, ndcg_at_k, recall_at_k


class TestRecallAtK:
    def test_all_relevant_chunks_found(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_half_the_relevant_chunks_found(self) -> None:
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == pytest.approx(0.5)

    def test_nothing_relevant_found(self) -> None:
        assert recall_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)

    def test_position_within_the_cutoff_does_not_matter(self) -> None:
        first = recall_at_k(["a", "x", "y"], {"a"}, k=3)
        last = recall_at_k(["x", "y", "a"], {"a"}, k=3)

        assert first == last == pytest.approx(1.0)

    def test_the_cutoff_is_enforced(self) -> None:
        # "a" sits at rank 3, outside K=2.
        assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == pytest.approx(0.0)

    def test_the_denominator_is_the_label_count_not_the_cutoff(self) -> None:
        # Three labels, only two can fit in the top 2: recall is capped at 2/3.
        assert recall_at_k(["a", "b"], {"a", "b", "c"}, k=2) == pytest.approx(2 / 3)

    def test_a_short_result_list_is_not_padded(self) -> None:
        assert recall_at_k(["a"], {"a", "b"}, k=5) == pytest.approx(0.5)

    def test_a_query_with_no_labels_is_rejected(self) -> None:
        # Returning 0.0 or 1.0 here would quietly bias the mean.
        with pytest.raises(ValueError, match="no relevant chunks"):
            recall_at_k(["a"], set(), k=3)

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            recall_at_k(["a"], {"a"}, k=0)


class TestDCG:
    def test_the_first_position_is_undiscounted(self) -> None:
        # log2(1 + 1) == 1
        assert dcg([1.0]) == pytest.approx(1.0)

    def test_each_position_is_discounted_by_log2_of_rank_plus_one(self) -> None:
        assert dcg([1.0, 1.0, 1.0]) == pytest.approx(
            1.0 + 1 / math.log2(3) + 1 / math.log2(4)
        )

    def test_zero_gains_contribute_nothing(self) -> None:
        assert dcg([0.0, 1.0]) == pytest.approx(1 / math.log2(3))

    def test_an_empty_sequence_scores_zero(self) -> None:
        assert dcg([]) == pytest.approx(0.0)


class TestNDCGAtK:
    def test_the_ideal_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == pytest.approx(1.0)

    def test_nothing_relevant_scores_zero(self) -> None:
        assert ndcg_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)

    def test_a_worse_ordering_scores_lower(self) -> None:
        good = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
        bad = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)

        assert good == pytest.approx(1.0)
        assert bad < good

    def test_the_value_matches_the_formula_by_hand(self) -> None:
        # One relevant chunk at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
        assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(1 / math.log2(3))

    def test_two_relevant_chunks_at_ranks_one_and_three(self) -> None:
        expected = (1.0 + 1 / math.log2(4)) / (1.0 + 1 / math.log2(3))

        assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(expected)

    def test_the_ideal_is_capped_at_k(self) -> None:
        # Three labels but K=1: the best any ranking can do is one hit, so a single
        # correct hit at rank 1 must score 1.0 rather than 1/3.
        assert ndcg_at_k(["a"], {"a", "b", "c"}, k=1) == pytest.approx(1.0)

    def test_the_score_stays_within_zero_and_one(self) -> None:
        for retrieved in (["a", "b"], ["b", "a"], ["x", "a"], ["x", "y"]):
            score = ndcg_at_k(retrieved, {"a", "b"}, k=2)
            assert 0.0 <= score <= 1.0

    def test_ranking_matters_where_recall_is_flat(self) -> None:
        # The reranker's job in one assertion: same recall, better nDCG.
        top = ["a", "x", "y"]
        bottom = ["x", "y", "a"]

        assert recall_at_k(top, {"a"}, k=3) == recall_at_k(bottom, {"a"}, k=3)
        assert ndcg_at_k(top, {"a"}, k=3) > ndcg_at_k(bottom, {"a"}, k=3)

    def test_a_query_with_no_labels_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no relevant chunks"):
            ndcg_at_k(["a"], set(), k=3)

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            ndcg_at_k(["a"], {"a"}, k=-1)
