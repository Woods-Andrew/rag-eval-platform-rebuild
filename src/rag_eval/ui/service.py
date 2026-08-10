"""The UI's view of the pipeline. Imports no UI framework.

Everything the Streamlit app needs to do lives here, behind plain function calls, so
the interesting behaviour — index once, build each retriever once, never re-embed on a
sidebar change — is testable without rendering anything.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..chunking import Chunker, FixedSizeChunker, StructureAwareChunker
from ..experiments import CorpusStats, describe_corpus
from ..factory import RETRIEVER_NAMES, RetrieverFactory
from ..generation import AnswerGenerator, GroundedAnswer, LanguageModel
from ..pipeline import build_corpus
from ..retrieval import Corpus, EmbeddingCache, RetrievalResult, Retriever, TextEncoder

__all__ = ["CHUNKERS", "RETRIEVERS", "RetrievalService", "SearchOutcome", "make_chunker"]

CHUNKERS = ("fixed", "structure")
RETRIEVERS = RETRIEVER_NAMES

EncoderFactory = Callable[[], TextEncoder]
RerankerFactory = Callable[[], object]
ModelFactory = Callable[[], LanguageModel]


def make_chunker(name: str) -> Chunker:
    """Build a chunker by name, rejecting anything not offered in the UI."""
    if name == "fixed":
        return FixedSizeChunker()
    if name == "structure":
        return StructureAwareChunker()
    raise ValueError(f"unknown chunker {name!r}; expected one of {', '.join(CHUNKERS)}")


@dataclass(frozen=True)
class SearchOutcome:
    """One search, plus the answer if one was asked for."""

    query: str
    retriever: str
    results: tuple[RetrievalResult, ...]
    answer: GroundedAnswer | None = None

    @property
    def found_anything(self) -> bool:
        return bool(self.results)


class RetrievalService:
    """An indexed document, and the retrievers built over it.

    Two things are cached for the life of the service, and both matter:

    * **The corpus.** Re-chunking on every interaction would be wasteful, but worse, it
      would re-derive chunk IDs on every rerun and make the UI's provenance unstable.
    * **Each retriever.** A ``DenseRetriever`` embeds the whole corpus when it is
      constructed. Rebuilding one because a radio button changed would re-embed the
      document on every click, which is precisely the mistake the "encode the corpus
      once" rule exists to prevent.

    Models arrive as factories rather than instances so nothing is downloaded until a
    strategy that needs one is actually selected — and so tests can pass fakes. An
    ``EmbeddingCache`` carries the corpus matrix across process restarts, so restarting
    the app does not re-embed a document it has already seen.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        chunker: Chunker | None = None,
        *,
        cache: EmbeddingCache | None = None,
        encoder_factory: EncoderFactory | None = None,
        reranker_factory: RerankerFactory | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.corpus: Corpus = build_corpus(pdf_path, chunker)
        self.stats: CorpusStats = describe_corpus(self.corpus)

        self._model_factory = model_factory
        self._factory = RetrieverFactory(
            self.corpus,
            cache=cache,
            encoder_factory=encoder_factory,
            reranker_factory=reranker_factory,
        )

    @property
    def document(self) -> str:
        return self.pdf_path.name

    @property
    def built_retrievers(self) -> tuple[str, ...]:
        """Which strategies have actually been constructed, in construction order."""
        return self._factory.built

    def retriever(self, name: str) -> Retriever:
        """Return the named retriever, building it once and reusing it after that."""
        return self._factory.get(name)

    def search(self, query: str, *, retriever: str, top_k: int) -> Sequence[RetrievalResult]:
        """Run one query. Validation lives in the retrievers, not duplicated here."""
        return self.retriever(retriever).retrieve(query, top_k=top_k)

    def ask(
        self, query: str, *, retriever: str, top_k: int, generate: bool = True
    ) -> SearchOutcome:
        """Retrieve, then optionally answer from exactly what was retrieved.

        The answer is built from the same results the UI displays — there is no second,
        hidden retrieval behind the generated text, so what a user reads and what the
        answer rests on are the same passages.
        """
        results = tuple(self.search(query, retriever=retriever, top_k=top_k))
        if not generate or self._model_factory is None:
            return SearchOutcome(query=query, retriever=retriever, results=results)

        generator = AnswerGenerator(self._model_factory(), max_evidence=top_k)
        return SearchOutcome(
            query=query,
            retriever=retriever,
            results=results,
            answer=generator.answer(query, results),
        )
