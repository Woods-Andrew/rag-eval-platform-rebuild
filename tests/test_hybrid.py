"""Hybrid retrieval: composition, candidate depth, and the lexical/dense complement."""

from __future__ import annotations

import pytest

from rag_eval.chunking import TextChunk
from rag_eval.retrieval import (
    BM25Retriever,
    Corpus,
    DenseRetriever,
    HybridRetriever,
    RetrievalResult,
    Retriever,
)
from tests.fakes import FakeEncoder


def chunk(chunk_id: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id, text=text, source="omics.pdf", page_number=1, chunk_index=0
    )


class ScriptedRetriever:
    """Returns a fixed ranking, recording the top_k it was asked for."""

    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunk_ids = chunk_ids
        self.requested_top_k: list[int] = []

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievalResult]:
        self.requested_top_k.append(top_k)
        return [
            RetrievalResult(chunk=chunk(chunk_id, chunk_id), score=1.0, rank=rank)
            for rank, chunk_id in enumerate(self.chunk_ids[:top_k], start=1)
        ]


class TestComposition:
    def test_satisfies_the_retriever_protocol(self) -> None:
        hybrid = HybridRetriever([ScriptedRetriever(["a"])])

        assert isinstance(hybrid, Retriever)

    def test_a_hybrid_of_hybrids_composes(self) -> None:
        inner = HybridRetriever([ScriptedRetriever(["a", "b"])])
        outer = HybridRetriever([inner, ScriptedRetriever(["b", "a"])])

        assert {result.chunk_id for result in outer.retrieve("q", top_k=2)} == {"a", "b"}

    def test_every_member_is_queried(self) -> None:
        first, second = ScriptedRetriever(["a"]), ScriptedRetriever(["b"])

        HybridRetriever([first, second]).retrieve("q", top_k=2)

        assert first.requested_top_k and second.requested_top_k

    def test_a_single_member_passes_its_ranking_through(self) -> None:
        hybrid = HybridRetriever([ScriptedRetriever(["a", "b", "c"])])

        assert [r.chunk_id for r in hybrid.retrieve("q", top_k=3)] == ["a", "b", "c"]


class TestCandidateDepth:
    def test_members_are_asked_for_more_than_the_caller_wants(self) -> None:
        # A chunk ranked 12th by one retriever and 3rd by the other is exactly the
        # case fusion exists for; truncating to top_k first would discard it.
        member = ScriptedRetriever(["a"])

        HybridRetriever([member], candidate_multiplier=4).retrieve("q", top_k=5)

        assert member.requested_top_k == [20]

    def test_the_multiplier_is_configurable(self) -> None:
        member = ScriptedRetriever(["a"])

        HybridRetriever([member], candidate_multiplier=1).retrieve("q", top_k=5)

        assert member.requested_top_k == [5]

    def test_a_deep_candidate_pool_can_change_the_fused_order(self) -> None:
        # "z" is second for both retrievers, so it is nobody's top hit but the only
        # chunk either one agrees on. At depth 1 neither reports it and the fused
        # winner is an arbitrary tie-break between two single-vote chunks; at depth 2
        # z picks up both terms (1/62 + 1/62) and overtakes them.
        lexical = ScriptedRetriever(["a", "z"])
        dense = ScriptedRetriever(["b", "z"])

        shallow = HybridRetriever([lexical, dense], candidate_multiplier=1)
        deep = HybridRetriever([lexical, dense], candidate_multiplier=2)

        assert shallow.retrieve("q", top_k=1)[0].chunk_id != "z"
        assert deep.retrieve("q", top_k=1)[0].chunk_id == "z"


class TestLexicalDenseComplement:
    @pytest.fixture
    def corpus(self) -> Corpus:
        return Corpus(
            [
                chunk("imputation", "missing modalities are imputed with a learned prior"),
                chunk("fusion", "reciprocal rank fusion combines ranked lists"),
                chunk("encoder", "the bi-encoder embeds query and passage separately"),
            ]
        )

    def test_hybrid_finds_what_bm25_alone_misses(self, corpus: Corpus) -> None:
        # The query paraphrases the imputation chunk without sharing a single term.
        query = "handling absent measurements"
        vectors = {
            "missing modalities are imputed with a learned prior": [1.0, 0.0],
            "reciprocal rank fusion combines ranked lists": [0.0, 1.0],
            "the bi-encoder embeds query and passage separately": [0.0, 1.0],
            query: [1.0, 0.0],
        }

        lexical = BM25Retriever(corpus)
        dense = DenseRetriever(corpus, FakeEncoder(vectors, dimension=2))
        hybrid = HybridRetriever([lexical, dense])

        assert lexical.retrieve(query, top_k=3) == []
        assert hybrid.retrieve(query, top_k=3)[0].chunk_id == "imputation"

    def test_hybrid_keeps_the_exact_lexical_match(self, corpus: Corpus) -> None:
        # Dense is deliberately unhelpful here; the lexical vote must still carry.
        query = "reciprocal rank fusion"
        lexical = BM25Retriever(corpus)
        dense = DenseRetriever(corpus, FakeEncoder({}, dimension=2))

        hybrid = HybridRetriever([lexical, dense])

        assert hybrid.retrieve(query, top_k=3)[0].chunk_id == "fusion"


class TestValidation:
    def test_at_least_one_member_is_required(self) -> None:
        with pytest.raises(ValueError, match="at least one member"):
            HybridRetriever([])

    def test_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            HybridRetriever([ScriptedRetriever(["a"])], k=0)

    def test_candidate_multiplier_must_be_at_least_one(self) -> None:
        with pytest.raises(ValueError, match="candidate_multiplier must be at least 1"):
            HybridRetriever([ScriptedRetriever(["a"])], candidate_multiplier=0)

    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_queries_are_rejected(self, query: str) -> None:
        with pytest.raises(ValueError, match="query must not be empty"):
            HybridRetriever([ScriptedRetriever(["a"])]).retrieve(query, top_k=3)

    def test_non_positive_top_k_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_k must be positive"):
            HybridRetriever([ScriptedRetriever(["a"])]).retrieve("q", top_k=0)
