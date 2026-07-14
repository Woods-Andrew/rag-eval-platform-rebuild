"""The retrieval data model: results, ranking, the corpus, and tokenization."""

from __future__ import annotations

import pytest

from rag_eval.chunking import TextChunk
from rag_eval.retrieval import (
    Corpus,
    RetrievalResult,
    rank_results,
    tokenize,
    validate_query,
    validate_top_k,
)


def chunk(chunk_id: str, text: str = "alpha beta", *, page: int = 1) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id, text=text, source="omics.pdf", page_number=page, chunk_index=0
    )


class TestRetrievalResult:
    def test_rank_is_one_indexed(self) -> None:
        with pytest.raises(ValueError, match="1-indexed"):
            RetrievalResult(chunk=chunk("a"), score=1.0, rank=0)

    def test_delegates_provenance_to_the_chunk(self) -> None:
        result = RetrievalResult(chunk=chunk("a", page=4), score=1.0, rank=1)

        assert result.chunk_id == "a"
        assert result.citation == "omics.pdf p.4"

    def test_results_are_immutable(self) -> None:
        result = RetrievalResult(chunk=chunk("a"), score=1.0, rank=1)

        with pytest.raises(AttributeError):
            result.score = 2.0  # type: ignore[misc]


class TestRankResults:
    def test_orders_by_descending_score(self) -> None:
        scored = [(chunk("a"), 0.1), (chunk("b"), 0.9), (chunk("c"), 0.5)]

        results = rank_results(scored, top_k=3)

        assert [result.chunk_id for result in results] == ["b", "c", "a"]

    def test_ranks_are_sequential_from_one(self) -> None:
        scored = [(chunk("a"), 0.1), (chunk("b"), 0.9), (chunk("c"), 0.5)]

        assert [result.rank for result in rank_results(scored, top_k=3)] == [1, 2, 3]

    def test_truncates_to_top_k(self) -> None:
        scored = [(chunk(name), score) for name, score in [("a", 0.1), ("b", 0.9), ("c", 0.5)]]

        results = rank_results(scored, top_k=2)

        assert [result.chunk_id for result in results] == ["b", "c"]

    def test_top_k_larger_than_the_corpus_returns_everything(self) -> None:
        assert len(rank_results([(chunk("a"), 1.0)], top_k=50)) == 1

    def test_ties_break_on_chunk_id_not_input_order(self) -> None:
        forwards = rank_results([(chunk("b"), 1.0), (chunk("a"), 1.0)], top_k=2)
        backwards = rank_results([(chunk("a"), 1.0), (chunk("b"), 1.0)], top_k=2)

        assert [result.chunk_id for result in forwards] == ["a", "b"]
        assert [result.chunk_id for result in forwards] == [r.chunk_id for r in backwards]

    def test_empty_input_yields_no_results(self) -> None:
        assert rank_results([], top_k=5) == []

    @pytest.mark.parametrize("top_k", [0, -1])
    def test_top_k_must_be_positive(self, top_k: int) -> None:
        with pytest.raises(ValueError, match="top_k must be positive"):
            rank_results([(chunk("a"), 1.0)], top_k=top_k)


class TestCorpus:
    def test_preserves_input_order(self) -> None:
        corpus = Corpus([chunk("b"), chunk("a")])

        assert corpus.chunk_ids == ("b", "a")

    def test_looks_chunks_up_by_id(self) -> None:
        corpus = Corpus([chunk("a"), chunk("b", "gamma")])

        assert corpus.get("b").text == "gamma"

    def test_unknown_id_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="no chunk with id 'missing'"):
            Corpus([chunk("a")]).get("missing")

    def test_duplicate_chunk_ids_are_rejected(self) -> None:
        # A duplicate ID means a relevance label points at two different passages.
        with pytest.raises(ValueError, match="duplicate chunk_id"):
            Corpus([chunk("a"), chunk("a", "different text")])

    def test_an_empty_corpus_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one chunk"):
            Corpus([])

    def test_supports_len_iteration_and_membership(self) -> None:
        corpus = Corpus([chunk("a"), chunk("b")])

        assert len(corpus) == 2
        assert [c.chunk_id for c in corpus] == ["a", "b"]
        assert "a" in corpus
        assert "z" not in corpus

    def test_sources_are_distinct_and_ordered(self) -> None:
        chunks = [
            TextChunk(chunk_id="a", text="x", source="one.pdf", page_number=1, chunk_index=0),
            TextChunk(chunk_id="b", text="x", source="two.pdf", page_number=1, chunk_index=0),
            TextChunk(chunk_id="c", text="x", source="one.pdf", page_number=2, chunk_index=0),
        ]

        assert Corpus(chunks).sources == ("one.pdf", "two.pdf")


class TestTokenize:
    def test_lowercases_and_drops_punctuation(self) -> None:
        assert tokenize("The Model, evaluated!") == ["the", "model", "evaluated"]

    def test_keeps_alphanumeric_terms_intact(self) -> None:
        assert tokenize("BM25 and nDCG@10") == ["bm25", "and", "ndcg", "10"]

    def test_hyphenated_compounds_yield_the_whole_and_the_parts(self) -> None:
        # So a query for "omics" still matches a document that only says "multi-omics".
        assert tokenize("multi-omics") == ["multi-omics", "multi", "omics"]

    def test_underscores_are_treated_as_joiners(self) -> None:
        assert tokenize("rag_eval") == ["rag_eval", "rag", "eval"]

    def test_empty_text_yields_no_tokens(self) -> None:
        assert tokenize("") == []
        assert tokenize("   !!!  ") == []


class TestValidation:
    def test_query_is_stripped(self) -> None:
        assert validate_query("  how are chunks scored?  ") == "how are chunks scored?"

    @pytest.mark.parametrize("query", ["", "   ", "\n\t"])
    def test_empty_queries_are_rejected(self, query: str) -> None:
        with pytest.raises(ValueError, match="query must not be empty"):
            validate_query(query)

    def test_top_k_must_be_positive(self) -> None:
        assert validate_top_k(5) == 5
        with pytest.raises(ValueError, match="top_k must be positive"):
            validate_top_k(0)
