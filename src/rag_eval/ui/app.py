"""The Streamlit interface. The only module in this project that imports Streamlit.

Everything below is presentation. The pipeline it drives lives in ``service.py`` and
knows nothing about the UI, which is the direction the dependency has to run: the
interface is a consumer of retrieval, never a dependency of it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import streamlit as st

from ..generation import GroundedAnswer, LanguageModel
from ..retrieval import EmbeddingCache, RetrievalResult
from .service import CHUNKERS, RETRIEVERS, RetrievalService, make_chunker

__all__ = ["main"]

DOCUMENTS = Path("data/documents")


@st.cache_resource(show_spinner="Indexing the document…")
def load_service(pdf_path: str, chunker: str) -> RetrievalService:
    """Index a document once per (document, chunking) pair.

    Streamlit reruns the whole script on every interaction, so without this the app
    would re-chunk and re-embed the corpus on each keystroke. The cache key is the pair
    because changing the chunker genuinely changes every chunk ID — that *is* a
    different index, not the same one viewed differently.
    """
    return RetrievalService(
        pdf_path,
        make_chunker(chunker),
        # Survives a restart, unlike the in-process cache above: relaunching the app
        # on a document it has already indexed reads the embeddings off disk.
        cache=EmbeddingCache(),
        model_factory=_model_factory(),
    )


def _model_factory() -> Callable[[], LanguageModel] | None:
    """Only offer generation when a key exists, rather than failing mid-question.

    Returns a factory, not a client: constructing one is deferred until a question is
    actually asked with generation enabled.
    """
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return None

    from ..generation import ClaudeLanguageModel

    return ClaudeLanguageModel


def main() -> None:
    st.set_page_config(page_title="RAG Evaluation Platform", page_icon="🔍", layout="wide")
    st.title("RAG Evaluation Platform")
    st.caption(
        "Hybrid retrieval over a technical document: BM25, dense embeddings, "
        "reciprocal rank fusion, and cross-encoder reranking."
    )

    documents = sorted(DOCUMENTS.glob("*.pdf")) if DOCUMENTS.is_dir() else []
    if not documents:
        _render_empty_state()
        return

    with st.sidebar:
        st.header("Configuration")
        document = st.selectbox("Document", documents, format_func=lambda path: path.name)
        chunker = st.radio("Chunking", CHUNKERS, help="Changing this rebuilds the index.")
        retriever = st.radio("Retrieval", RETRIEVERS, index=RETRIEVERS.index("hybrid"))
        top_k = st.slider("Passages (K)", min_value=1, max_value=20, value=5)
        generate = st.checkbox(
            "Generate an answer",
            value=False,
            help="Requires ANTHROPIC_API_KEY. Retrieval works without it.",
        )

    service = load_service(str(document), chunker)
    _render_corpus_stats(service)

    query = st.text_input("Question", placeholder="how are missing modalities handled?")
    if not query.strip():
        return

    if generate and _model_factory() is None:
        st.warning("No ANTHROPIC_API_KEY set — showing retrieval only.")
        generate = False

    with st.spinner("Retrieving…"):
        outcome = service.ask(query, retriever=retriever, top_k=top_k, generate=generate)

    if not outcome.found_anything:
        st.info(f"No passage in this document matched that query under {retriever}.")
        return

    if outcome.answer is not None:
        _render_answer(outcome.answer)

    _render_results(outcome.results)


def _render_empty_state() -> None:
    st.info(
        f"No PDFs found in `{DOCUMENTS}/`. Add a document there to index it.\n\n"
        "Source documents are gitignored; the benchmark labels are version controlled."
    )


def _render_corpus_stats(service: RetrievalService) -> None:
    stats = service.stats
    columns = st.columns(4)
    columns[0].metric("Chunks", stats.chunk_count)
    columns[1].metric("Pages", stats.page_count)
    columns[2].metric("Median words", f"{stats.median_words:.0f}")
    columns[3].metric("Longest chunk", f"{stats.max_words} words")


def _render_answer(answer: GroundedAnswer) -> None:
    st.subheader("Answer")
    if not answer.has_answer:
        st.warning(answer.text)
        return

    st.markdown(answer.text)

    if answer.unresolved_citations:
        markers = ", ".join(f"[{marker}]" for marker in answer.unresolved_citations)
        # Surfaced, never hidden: a citation pointing at a passage that was never
        # supplied is the failure this whole pipeline exists to make visible.
        st.error(f"The answer cited {markers}, which was never supplied as evidence.")

    if answer.citations:
        st.caption(
            "Sources: "
            + " · ".join(f"[{item.marker}] {item.label}" for item in answer.citations)
        )


def _render_results(results: tuple[RetrievalResult, ...]) -> None:
    st.subheader(f"Retrieved passages ({len(results)})")
    for result in results:
        with st.expander(f"{result.rank}. {result.citation}  ·  score {result.score:.4f}"):
            st.code(result.chunk_id, language=None)
            st.write(result.chunk.text)


if __name__ == "__main__":  # pragma: no cover
    main()
