"""What generation consumes, and what it produces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..retrieval import RetrievalResult

__all__ = ["Citation", "Evidence", "GroundedAnswer", "evidence_from_results"]


@dataclass(frozen=True)
class Evidence:
    """One retrieved chunk, numbered for the model to cite.

    The marker is the model's only handle on a passage. It is 1-indexed because models
    cite ``[1]`` far more reliably than ``[0]``, and because a marker that reads as an
    ordinal is easier to check by eye against the rendered evidence list.
    """

    marker: int
    result: RetrievalResult

    def __post_init__(self) -> None:
        if self.marker < 1:
            raise ValueError(f"evidence markers are 1-indexed, got {self.marker}")

    @property
    def chunk_id(self) -> str:
        return self.result.chunk_id

    @property
    def text(self) -> str:
        return self.result.chunk.text

    @property
    def citation(self) -> str:
        return self.result.citation

    def to_citation(self) -> Citation:
        chunk = self.result.chunk
        return Citation(
            marker=self.marker,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            page_number=chunk.page_number,
            section=chunk.section,
        )


@dataclass(frozen=True)
class Citation:
    """A resolved reference from an answer back to the page it came from.

    This is the end of the provenance chain that started at PDF extraction: source,
    page, and chunk ID all survive to the citation, so any sentence in an answer can be
    traced to a specific passage on a specific page.
    """

    marker: int
    chunk_id: str
    source: str
    page_number: int
    section: str | None = None

    @property
    def label(self) -> str:
        """Human-readable provenance, e.g. ``omics.pdf p.3 § Methods``."""
        base = f"{self.source} p.{self.page_number}"
        return f"{base} § {self.section}" if self.section else base


@dataclass(frozen=True)
class GroundedAnswer:
    """An answer, the citations it resolved, and the evidence it was given.

    ``citations`` holds only markers that resolved to real evidence. Markers the model
    invented land in ``unresolved_citations`` rather than being dropped: a citation
    pointing at a passage that was never supplied is the failure mode this whole
    pipeline exists to make visible, so it is reported, not swallowed.
    """

    query: str
    text: str
    evidence: tuple[Evidence, ...]
    citations: tuple[Citation, ...] = ()
    unresolved_citations: tuple[int, ...] = ()
    has_answer: bool = True

    @property
    def is_grounded(self) -> bool:
        """True when the answer cites at least one real passage and invents none."""
        return self.has_answer and bool(self.citations) and not self.unresolved_citations

    @property
    def cited_pages(self) -> tuple[tuple[str, int], ...]:
        """Distinct ``(source, page)`` pairs the answer rests on, in citation order."""
        return tuple(
            dict.fromkeys((citation.source, citation.page_number) for citation in self.citations)
        )

    def format(self) -> str:
        """The answer followed by its sources, the way a reader wants to see it."""
        if not self.citations:
            return self.text
        sources = "\n".join(
            f"  [{citation.marker}] {citation.label}" for citation in self.citations
        )
        return f"{self.text}\n\nsources:\n{sources}"


def evidence_from_results(results: Sequence[RetrievalResult]) -> tuple[Evidence, ...]:
    """Number retrieved results for citation, preserving retrieval order.

    Order is preserved rather than re-sorted: the retriever already decided what is most
    relevant, and generation consumes that decision instead of second-guessing it.
    """
    return tuple(
        Evidence(marker=marker, result=result) for marker, result in enumerate(results, start=1)
    )
