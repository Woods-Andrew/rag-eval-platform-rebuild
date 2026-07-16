"""Dense retrieval: cosine ranking, encode-once, and the encoder boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rag_eval.chunking import TextChunk
from rag_eval.retrieval import Corpus, DenseRetriever, Retriever, TextEncoder, l2_normalize
from tests.fakes import FakeEncoder


def chunk(chunk_id: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id, text=text, source="omics.pdf", page_number=1, chunk_index=0
    )


# Three orthogonal-ish directions, so the expected ranking is obvious by eye.
VECTORS = {
    "imputation of missing modalities": [1.0, 0.0, 0.0],
    "reciprocal rank fusion": [0.0, 1.0, 0.0],
    "cross-encoder reranking": [0.0, 0.0, 1.0],
    "handling absent measurements": [0.9, 0.1, 0.0],  # paraphrase of the first chunk
    "how are ranked lists combined": [0.1, 0.9, 0.0],
}


@pytest.fixture
def corpus() -> Corpus:
    return Corpus(
        [
            chunk("imputation", "imputation of missing modalities"),
            chunk("fusion", "reciprocal rank fusion"),
            chunk("rerank", "cross-encoder reranking"),
        ]
    )


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder(VECTORS, dimension=3)


@pytest.fixture
def retriever(corpus: Corpus, encoder: FakeEncoder) -> DenseRetriever:
    return DenseRetriever(corpus, encoder)


class TestRanking:
    def test_the_nearest_vector_ranks_first(self, retriever: DenseRetriever) -> None:
        results = retriever.retrieve("reciprocal rank fusion", top_k=3)

        assert results[0].chunk_id == "fusion"

    def test_paraphrase_is_matched_where_bm25_would_fail(
        self, retriever: DenseRetriever
    ) -> None:
        # Shares no vocabulary with "imputation of missing modalities".
        results = retriever.retrieve("handling absent measurements", top_k=3)

        assert results[0].chunk_id == "imputation"

    def test_scores_are_cosine_similarities(self, retriever: DenseRetriever) -> None:
        results = retriever.retrieve("reciprocal rank fusion", top_k=3)

        by_id = {result.chunk_id: result.score for result in results}
        assert by_id["fusion"] == pytest.approx(1.0)
        assert by_id["rerank"] == pytest.approx(0.0)

    def test_magnitude_does_not_affect_ranking(self, corpus: Corpus) -> None:
        # Same directions, wildly different lengths: normalization must cancel them.
        scaled = FakeEncoder(
            {text: [value * 100 for value in vector] for text, vector in VECTORS.items()},
            dimension=3,
        )

        baseline = DenseRetriever(corpus, FakeEncoder(VECTORS, dimension=3))
        stretched = DenseRetriever(corpus, scaled)

        query = "how are ranked lists combined"
        assert [r.chunk_id for r in baseline.retrieve(query, top_k=3)] == [
            r.chunk_id for r in stretched.retrieve(query, top_k=3)
        ]

    def test_scores_are_non_increasing(self, retriever: DenseRetriever) -> None:
        scores = [r.score for r in retriever.retrieve("how are ranked lists combined", top_k=3)]

        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates(self, retriever: DenseRetriever) -> None:
        assert len(retriever.retrieve("reciprocal rank fusion", top_k=2)) == 2

    def test_an_unknown_query_scores_zero_against_everything(
        self, retriever: DenseRetriever
    ) -> None:
        results = retriever.retrieve("entirely unseen text", top_k=3)

        assert all(result.score == pytest.approx(0.0) for result in results)

    def test_results_are_deterministic(self, retriever: DenseRetriever) -> None:
        first = [r.chunk_id for r in retriever.retrieve("cross-encoder reranking", top_k=3)]
        second = [r.chunk_id for r in retriever.retrieve("cross-encoder reranking", top_k=3)]

        assert first == second


class TestEncodeOnce:
    def test_the_corpus_is_encoded_exactly_once_at_construction(
        self, corpus: Corpus, encoder: FakeEncoder
    ) -> None:
        DenseRetriever(corpus, encoder)

        assert len(encoder.calls) == 1
        assert encoder.encoded_texts == [c.text for c in corpus]

    def test_queries_never_re_encode_the_corpus(
        self, corpus: Corpus, encoder: FakeEncoder
    ) -> None:
        retriever = DenseRetriever(corpus, encoder)
        for _ in range(5):
            retriever.retrieve("reciprocal rank fusion", top_k=2)

        # One call to build the index, then exactly one single-text call per query.
        assert len(encoder.calls) == 6
        assert all(len(batch) == 1 for batch in encoder.calls[1:])


class TestEncoderBoundary:
    def test_satisfies_the_retriever_protocol(self, retriever: DenseRetriever) -> None:
        assert isinstance(retriever, Retriever)

    def test_the_fake_satisfies_the_encoder_protocol(self, encoder: FakeEncoder) -> None:
        assert isinstance(encoder, TextEncoder)

    def test_dimension_is_taken_from_the_encoder(self, retriever: DenseRetriever) -> None:
        assert retriever.dimension == 3

    def test_a_wrong_vector_count_is_rejected(self, corpus: Corpus) -> None:
        class TooFew:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.ones((1, 3), dtype=np.float32)

        with pytest.raises(ValueError, match="1 vectors for 3 chunks"):
            DenseRetriever(corpus, TooFew())

    def test_a_one_dimensional_return_is_rejected(self, corpus: Corpus) -> None:
        class Flat:
            def encode(self, texts: list[str]) -> np.ndarray:
                return np.ones(3, dtype=np.float32)

        with pytest.raises(ValueError, match="2-D array"):
            DenseRetriever(corpus, Flat())

    def test_importing_retrieval_does_not_import_sentence_transformers(self) -> None:
        # The heavy import is deferred into SentenceTransformerEncoder.__init__ so the
        # offline suite never pays for torch and cannot reach the network by accident.
        # A subprocess is the only honest way to check: this process has already
        # imported whatever the rest of the suite needed.
        import rag_eval

        source_root = Path(rag_eval.__file__).resolve().parents[1]
        code = (
            "import sys; import rag_eval.retrieval; "
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


class TestNormalization:
    def test_rows_become_unit_length(self) -> None:
        normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))

        assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)

    def test_zero_rows_are_left_alone_rather_than_dividing_by_zero(self) -> None:
        normalized = l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32))

        assert np.allclose(normalized[0], 0.0)
        assert np.isfinite(normalized).all()


class TestValidation:
    @pytest.mark.parametrize("query", ["", "  "])
    def test_empty_queries_are_rejected(self, retriever: DenseRetriever, query: str) -> None:
        with pytest.raises(ValueError, match="query must not be empty"):
            retriever.retrieve(query, top_k=3)

    def test_non_positive_top_k_is_rejected(self, retriever: DenseRetriever) -> None:
        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.retrieve("reciprocal rank fusion", top_k=0)
