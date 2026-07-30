"""The worked examples in docs/methodology.md, checked against the implementation.

Documentation that states numbers can rot silently. These recompute every worked figure
and assert the documented value is still what the code produces — the same standard
applied to benchmark results, applied to the maths that explains them.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rag_eval.chunking import TextChunk
from rag_eval.evaluation import ndcg_at_k, recall_at_k
from rag_eval.retrieval import RetrievalResult, rrf_scores

DOCS = Path(__file__).resolve().parents[1] / "docs"
METHODOLOGY = DOCS / "methodology.md"


def ranked(chunk_ids: list[str]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk=TextChunk(
                chunk_id=chunk_id,
                text=chunk_id,
                source="paper.pdf",
                page_number=1,
                chunk_index=0,
            ),
            score=1.0,
            rank=rank,
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


@pytest.fixture(scope="module")
def methodology() -> str:
    return METHODOLOGY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fused() -> dict[str, float]:
    """The worked fusion example: BM25 [a, b, z] against dense [z, a, b]."""
    return rrf_scores([ranked(["a", "b", "z"]), ranked(["z", "a", "b"])])


class TestDocsExist:
    @pytest.mark.parametrize(
        "name", ["architecture.md", "methodology.md", "decisions.md"]
    )
    def test_the_reference_documents_are_present(self, name: str) -> None:
        assert (DOCS / name).is_file()


class TestWorkedRecall:
    def test_the_documented_recall_is_what_the_code_computes(self) -> None:
        assert recall_at_k(["b", "a", "d", "e", "f"], {"a", "c"}, 5) == pytest.approx(0.5)

    def test_the_denominator_is_the_label_count_not_k(self) -> None:
        # The doc claims two of three relevant chunks scores 0.667 regardless of K.
        assert recall_at_k(["a", "b", "x"], {"a", "b", "z"}, 3) == pytest.approx(2 / 3, abs=5e-4)
        assert recall_at_k(["a", "b", "x"], {"a", "b", "z"}, 50) == pytest.approx(2 / 3, abs=5e-4)

    def test_the_worked_value_appears_in_the_document(self, methodology: str) -> None:
        assert "`1/2 = 0.5`" in methodology


class TestWorkedNdcg:
    RETRIEVED = ["b", "a", "d", "c", "e"]
    RELEVANT = {"a", "c"}

    def test_the_documented_ndcg_is_what_the_code_computes(self) -> None:
        assert ndcg_at_k(self.RETRIEVED, self.RELEVANT, 5) == pytest.approx(0.651, abs=5e-4)

    def test_the_documented_dcg_and_idcg_still_hold(self) -> None:
        computed_dcg = 1 / math.log2(3) + 1 / math.log2(5)
        computed_idcg = 1 / math.log2(2) + 1 / math.log2(3)

        assert computed_dcg == pytest.approx(1.062, abs=5e-4)
        assert computed_idcg == pytest.approx(1.631, abs=5e-4)
        assert computed_dcg / computed_idcg == pytest.approx(0.651, abs=5e-4)

    @pytest.mark.parametrize(
        ("rank", "discount"), [(1, 1.000), (2, 0.631), (3, 0.500), (4, 0.431), (5, 0.387)]
    )
    def test_every_discount_in_the_table_is_correct(self, rank: int, discount: float) -> None:
        assert 1 / math.log2(rank + 1) == pytest.approx(discount, abs=5e-4)

    def test_idcg_is_capped_at_k(self) -> None:
        # Five relevant chunks at K=3: a perfect top-3 must still score exactly 1.0.
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c", "d", "e"}, 3) == pytest.approx(1.0)

    def test_the_worked_values_appear_in_the_document(self, methodology: str) -> None:
        assert "`DCG@5 = 1.062`" in methodology
        assert "`IDCG@5 = 1.000 + 0.631 = 1.631`" in methodology
        assert "`nDCG@5 = 1.062 / 1.631 = 0.651`" in methodology


class TestWorkedFusion:
    @pytest.mark.parametrize(
        ("chunk_id", "documented"), [("a", 0.03252), ("z", 0.03227), ("b", 0.03200)]
    )
    def test_each_documented_fusion_score_is_correct(
        self, fused: dict[str, float], chunk_id: str, documented: float
    ) -> None:
        assert fused[chunk_id] == pytest.approx(documented, abs=5e-6)

    def test_the_documented_winner_still_wins(self, fused: dict[str, float]) -> None:
        # Consistently near the top of both beats first-in-one, third-in-the-other.
        assert max(fused, key=lambda key: fused[key]) == "a"

    def test_the_documented_ordering_holds(self, fused: dict[str, float]) -> None:
        assert sorted(fused, key=lambda key: -fused[key]) == ["a", "z", "b"]

    def test_the_scores_appear_in_the_document(self, methodology: str) -> None:
        for documented in ("0.03252", "0.03227", "0.03200"):
            assert documented in methodology


class TestWorkedBm25Degeneracy:
    def test_okapi_idf_is_exactly_zero_at_two_documents(self) -> None:
        # The claim that makes an empty BM25 result set unsurprising.
        assert math.log(2 - 1 + 0.5) - math.log(1 + 0.5) == 0.0

    def test_the_claim_appears_in_the_document(self, methodology: str) -> None:
        assert "log(1.5) − log(1.5) = 0" in methodology
