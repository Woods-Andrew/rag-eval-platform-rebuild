"""BM25 retrieval: ranking, lexical strengths, and its known blind spot."""

from __future__ import annotations

import pytest

from rag_eval.chunking import TextChunk
from rag_eval.retrieval import BM25Retriever, Corpus, Retriever


def chunk(chunk_id: str, text: str, *, page: int = 1) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id, text=text, source="omics.pdf", page_number=page, chunk_index=0
    )


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        [
            chunk("imputation", "Missing modalities are handled by a learned imputation prior."),
            chunk("fusion", "Reciprocal rank fusion combines two ranked lists into one."),
            chunk("encoder", "The bi-encoder embeds the query and passage separately."),
            chunk("omics", "The multi-omics embedding is disease-aware and adaptive."),
        ]
    )


@pytest.fixture
def retriever(corpus: Corpus) -> BM25Retriever:
    return BM25Retriever(corpus)


class TestRanking:
    def test_the_chunk_containing_the_query_terms_ranks_first(
        self, retriever: BM25Retriever
    ) -> None:
        results = retriever.retrieve("reciprocal rank fusion", top_k=3)

        assert results[0].chunk_id == "fusion"

    def test_ranks_are_sequential_from_one(self, retriever: BM25Retriever) -> None:
        results = retriever.retrieve("embedding query passage", top_k=3)

        assert [result.rank for result in results] == list(range(1, len(results) + 1))

    def test_scores_are_non_increasing(self, retriever: BM25Retriever) -> None:
        results = retriever.retrieve("the query embedding", top_k=4)

        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_the_number_of_results(self, retriever: BM25Retriever) -> None:
        assert len(retriever.retrieve("the", top_k=1)) <= 1

    def test_results_are_deterministic_across_calls(self, retriever: BM25Retriever) -> None:
        first = [r.chunk_id for r in retriever.retrieve("encoder passage", top_k=4)]
        second = [r.chunk_id for r in retriever.retrieve("encoder passage", top_k=4)]

        assert first == second


class TestLexicalBehaviour:
    def test_a_query_with_no_overlapping_terms_returns_nothing(
        self, retriever: BM25Retriever
    ) -> None:
        # Padding the list with zero-scoring chunks would inflate Recall@K.
        assert retriever.retrieve("quantum chromodynamics tractor", top_k=5) == []

    def test_matching_is_case_insensitive(self, retriever: BM25Retriever) -> None:
        assert retriever.retrieve("IMPUTATION", top_k=1)[0].chunk_id == "imputation"

    def test_a_hyphenated_compound_is_found_by_one_of_its_parts(
        self, retriever: BM25Retriever
    ) -> None:
        # "omics" alone appears nowhere in the corpus except inside "multi-omics".
        assert retriever.retrieve("omics", top_k=1)[0].chunk_id == "omics"

    def test_paraphrase_is_not_matched(self, retriever: BM25Retriever) -> None:
        # BM25's documented blind spot, and the reason dense retrieval is added:
        # "handling absent data" shares no term with "missing modalities".
        results = retriever.retrieve("strategy for absent measurements", top_k=4)

        assert [result.chunk_id for result in results] != ["imputation"]

    def test_results_carry_full_provenance(self, retriever: BM25Retriever) -> None:
        result = retriever.retrieve("reciprocal rank fusion", top_k=1)[0]

        assert result.citation == "omics.pdf p.1"
        assert result.chunk.text.startswith("Reciprocal rank fusion")


class TestConstruction:
    def test_satisfies_the_retriever_protocol(self, retriever: BM25Retriever) -> None:
        assert isinstance(retriever, Retriever)

    def test_a_custom_tokenizer_is_injectable(self, corpus: Corpus) -> None:
        calls: list[str] = []

        def shouting_tokenizer(text: str) -> list[str]:
            calls.append(text)
            return text.lower().split()

        BM25Retriever(corpus, tokenizer=shouting_tokenizer).retrieve("fusion", top_k=1)

        assert "fusion" in calls

    def test_a_corpus_with_no_searchable_text_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no searchable text"):
            BM25Retriever(Corpus([chunk("a", "...")]), tokenizer=lambda _: [])

    def test_the_corpus_is_tokenized_once_not_per_query(self, corpus: Corpus) -> None:
        calls: list[str] = []

        def counting_tokenizer(text: str) -> list[str]:
            calls.append(text)
            return text.lower().split()

        retriever = BM25Retriever(corpus, tokenizer=counting_tokenizer)
        after_build = len(calls)
        for _ in range(3):
            retriever.retrieve("fusion", top_k=2)

        assert after_build == len(corpus)
        assert len(calls) == after_build + 3  # one per query, none for the corpus


class TestValidation:
    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_queries_are_rejected(self, retriever: BM25Retriever, query: str) -> None:
        with pytest.raises(ValueError, match="query must not be empty"):
            retriever.retrieve(query, top_k=5)

    @pytest.mark.parametrize("top_k", [0, -3])
    def test_non_positive_top_k_is_rejected(
        self, retriever: BM25Retriever, top_k: int
    ) -> None:
        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("fusion", top_k=top_k)
