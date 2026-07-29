"""The embedding cache: what it saves, and what it refuses to reuse."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from rag_eval.chunking import FixedSizeChunker, StructureAwareChunker
from rag_eval.factory import RetrieverFactory
from rag_eval.pipeline import build_corpus
from rag_eval.retrieval import (
    Corpus,
    DenseRetriever,
    EmbeddingCache,
    encoder_identity,
    fingerprint,
)
from tests.fakes import FakeEncoder, FakeReranker

PdfFactory = Callable[..., Path]

PAGES = [
    "Introduction\nThe multi-omics embedding is disease aware and adaptive for patients.",
    "Methods\nMissing modalities are imputed with a learned prior across cohorts.",
    "Results\nThe adaptive gate improved downstream survival prediction accuracy.",
    "Discussion\nAblations show the disease aware routing contributes most of the gain.",
    "Related Work\nEarlier fusion approaches concatenate modality features naively.",
    "Conclusion\nAn agentic framework for adaptive multi-omics embedding is presented.",
    "Appendix\nHyperparameters were selected by grid search over the validation split.",
]


@pytest.fixture
def paper(make_pdf: PdfFactory) -> Path:
    return make_pdf(PAGES, name="paper.pdf")


@pytest.fixture
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "embeddings")


class NamedEncoder(FakeEncoder):
    """A fake that reports a model name, the way the real encoder does."""

    def __init__(self, model_name: str = "fake-model", *, dimension: int = 4) -> None:
        super().__init__({}, dimension=dimension)
        self.model_name = model_name


def corpus_for(paper: Path, chunker: object = None) -> Corpus:
    return build_corpus(paper, chunker)  # type: ignore[arg-type]


class TestFingerprint:
    def test_the_same_model_and_chunks_give_the_same_key(self) -> None:
        assert fingerprint("m", ["a", "b"]) == fingerprint("m", ["a", "b"])

    def test_a_different_model_gives_a_different_key(self) -> None:
        assert fingerprint("m", ["a", "b"]) != fingerprint("other", ["a", "b"])

    def test_different_chunks_give_a_different_key(self) -> None:
        assert fingerprint("m", ["a", "b"]) != fingerprint("m", ["a", "c"])

    def test_reordering_the_corpus_gives_a_different_key(self) -> None:
        # Row order carries meaning: row i is chunk i.
        assert fingerprint("m", ["a", "b"]) != fingerprint("m", ["b", "a"])

    def test_ids_cannot_run_together_into_the_same_key(self) -> None:
        # Without a separator, ["ab", "c"] and ["a", "bc"] would hash identically.
        assert fingerprint("m", ["ab", "c"]) != fingerprint("m", ["a", "bc"])

    def test_an_encoder_is_identified_by_its_model_name(self) -> None:
        assert encoder_identity(NamedEncoder("all-MiniLM")) == "all-MiniLM"

    def test_an_encoder_without_a_model_name_falls_back_to_its_class(self) -> None:
        assert encoder_identity(FakeEncoder({}, dimension=2)) == "FakeEncoder"


class TestCacheRoundTrip:
    def test_a_cold_cache_is_a_miss(self, cache: EmbeddingCache) -> None:
        assert cache.load(fingerprint("m", ["a"]), ["a"]) is None

    def test_stored_embeddings_come_back_unchanged(self, cache: EmbeddingCache) -> None:
        embeddings = np.arange(8, dtype=np.float32).reshape(2, 4)
        key = fingerprint("m", ["a", "b"])

        cache.store(key, ["a", "b"], embeddings)

        assert np.array_equal(cache.load(key, ["a", "b"]), embeddings)

    def test_the_cache_directory_is_created_on_demand(self, tmp_path: Path) -> None:
        nested = EmbeddingCache(tmp_path / "deep" / "nested")

        written = nested.store("k", ["a"], np.zeros((1, 2), dtype=np.float32))

        assert written is not None and written.is_file()

    def test_a_mismatched_row_count_is_refused_at_write_time(
        self, cache: EmbeddingCache
    ) -> None:
        with pytest.raises(ValueError, match="cannot cache 2 vectors for 3 chunks"):
            cache.store("k", ["a", "b", "c"], np.zeros((2, 4), dtype=np.float32))

    def test_no_temporary_file_is_left_behind(self, cache: EmbeddingCache) -> None:
        cache.store("k", ["a"], np.zeros((1, 2), dtype=np.float32))

        assert list(Path(cache.directory).glob("*.tmp")) == []


class TestCacheInvalidation:
    def test_different_chunk_ids_are_a_miss_even_under_the_same_key(
        self, cache: EmbeddingCache
    ) -> None:
        # The guard that makes the cache safe: the key could collide, the stored IDs
        # cannot silently disagree.
        cache.store("k", ["a", "b"], np.zeros((2, 4), dtype=np.float32))

        assert cache.load("k", ["a", "c"]) is None

    def test_a_shorter_corpus_is_a_miss(self, cache: EmbeddingCache) -> None:
        cache.store("k", ["a", "b"], np.zeros((2, 4), dtype=np.float32))

        assert cache.load("k", ["a"]) is None

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, cache: EmbeddingCache) -> None:
        cache.store("k", ["a"], np.zeros((1, 4), dtype=np.float32))
        cache.path_for("k").write_bytes(b"not an npz file")

        assert cache.load("k", ["a"]) is None

    def test_an_unwritable_directory_degrades_to_no_caching(self, tmp_path: Path) -> None:
        blocked = tmp_path / "blocked"
        blocked.write_text("this is a file, not a directory")

        stored = EmbeddingCache(blocked).store("k", ["a"], np.zeros((1, 2), dtype=np.float32))

        assert stored is None


class TestDenseRetrieverCaching:
    def test_the_first_construction_encodes_and_the_second_does_not(
        self, paper: Path, cache: EmbeddingCache
    ) -> None:
        corpus = corpus_for(paper)
        first = NamedEncoder()
        second = NamedEncoder()

        cold = DenseRetriever(corpus, first, cache=cache)
        warm = DenseRetriever(corpus, second, cache=cache)

        assert cold.loaded_from_cache is False
        assert warm.loaded_from_cache is True
        assert first.calls  # the corpus was embedded once
        assert second.calls == []  # and never again

    def test_cached_and_uncached_retrievers_rank_identically(
        self, paper: Path, cache: EmbeddingCache
    ) -> None:
        # The cache must be invisible in the results, or it is not a cache.
        corpus = corpus_for(paper)
        vectors = {chunk.text: np.random.default_rng(0).normal(size=4) for chunk in corpus}
        vectors["imputed prior"] = np.random.default_rng(1).normal(size=4)

        uncached = DenseRetriever(corpus, FakeEncoder(vectors, dimension=4))
        DenseRetriever(corpus, FakeEncoder(vectors, dimension=4), cache=cache)
        warm = DenseRetriever(corpus, FakeEncoder(vectors, dimension=4), cache=cache)

        assert [r.chunk_id for r in warm.retrieve("imputed prior", top_k=5)] == [
            r.chunk_id for r in uncached.retrieve("imputed prior", top_k=5)
        ]

    def test_the_query_is_still_encoded_on_a_cache_hit(
        self, paper: Path, cache: EmbeddingCache
    ) -> None:
        # Only the corpus is cached; a query has never been seen before.
        corpus = corpus_for(paper)
        DenseRetriever(corpus, NamedEncoder(), cache=cache)
        encoder = NamedEncoder()

        DenseRetriever(corpus, encoder, cache=cache).retrieve("a query", top_k=3)

        assert encoder.encoded_texts == ["a query"]

    def test_a_different_chunker_does_not_reuse_the_cache(
        self, paper: Path, cache: EmbeddingCache
    ) -> None:
        # Re-chunking changes every chunk ID, so the old matrix describes other text.
        DenseRetriever(corpus_for(paper, FixedSizeChunker()), NamedEncoder(), cache=cache)
        encoder = NamedEncoder()

        DenseRetriever(
            corpus_for(paper, StructureAwareChunker(min_words=1)), encoder, cache=cache
        )

        assert encoder.calls

    def test_a_different_model_does_not_reuse_the_cache(
        self, paper: Path, cache: EmbeddingCache
    ) -> None:
        corpus = corpus_for(paper)
        DenseRetriever(corpus, NamedEncoder("model-a"), cache=cache)
        encoder = NamedEncoder("model-b")

        DenseRetriever(corpus, encoder, cache=cache)

        assert encoder.calls

    def test_without_a_cache_nothing_is_written(
        self, paper: Path, tmp_path: Path
    ) -> None:
        directory = tmp_path / "embeddings"

        DenseRetriever(corpus_for(paper), NamedEncoder())

        assert not directory.exists()


class TestFactorySharing:
    def test_the_encoder_is_constructed_once_across_every_strategy(
        self, paper: Path
    ) -> None:
        calls: list[int] = []

        def encoder_factory() -> NamedEncoder:
            calls.append(1)
            return NamedEncoder()

        factory = RetrieverFactory(
            corpus_for(paper),
            encoder_factory=encoder_factory,
            reranker_factory=lambda: FakeReranker({}),
        )
        factory.all()

        assert len(calls) == 1

    def test_the_corpus_is_embedded_once_across_every_strategy(self, paper: Path) -> None:
        encoder = NamedEncoder()
        factory = RetrieverFactory(
            corpus_for(paper),
            encoder_factory=lambda: encoder,
            reranker_factory=lambda: FakeReranker({}),
        )

        factory.all()

        assert len(encoder.calls) == 1

    def test_nothing_is_built_before_it_is_asked_for(self, paper: Path) -> None:
        factory = RetrieverFactory(corpus_for(paper), encoder_factory=NamedEncoder)

        assert factory.built == ()

    def test_bm25_alone_never_constructs_an_encoder(self, paper: Path) -> None:
        def explode() -> NamedEncoder:
            raise AssertionError("bm25 must not construct an encoder")

        RetrieverFactory(corpus_for(paper), encoder_factory=explode).get("bm25")

    def test_an_unknown_strategy_is_rejected(self, paper: Path) -> None:
        with pytest.raises(ValueError, match="unknown retriever"):
            RetrieverFactory(corpus_for(paper)).get("magic")

    def test_a_retriever_is_returned_identically_on_repeat_calls(self, paper: Path) -> None:
        factory = RetrieverFactory(corpus_for(paper))

        assert factory.get("bm25") is factory.get("bm25")

    def test_the_cache_reaches_the_dense_retriever(
        self, paper: Path, cache: EmbeddingCache
    ) -> None:
        corpus = corpus_for(paper)
        RetrieverFactory(corpus, cache=cache, encoder_factory=NamedEncoder).get("dense")

        warm = RetrieverFactory(corpus, cache=cache, encoder_factory=NamedEncoder).get("dense")

        assert warm.loaded_from_cache is True  # type: ignore[union-attr]
