"""Turning retrieved evidence into a cited answer."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..retrieval import RetrievalResult
from .llm import GenerationError, LanguageModel
from .models import Citation, Evidence, GroundedAnswer, evidence_from_results
from .prompt import INSUFFICIENT_EVIDENCE, NO_EVIDENCE_TEXT, SYSTEM_PROMPT, build_prompt

__all__ = ["DEFAULT_MAX_EVIDENCE", "AnswerGenerator"]

DEFAULT_MAX_EVIDENCE = 5

# Matches [2], [1][3], and [1, 3] — the three shapes models actually produce.
_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


class AnswerGenerator:
    """Answers a question from passages a retriever already selected.

    Generation is strictly downstream: it consumes results and never triggers or
    re-runs retrieval. That is what keeps retrieval measurable on its own terms — if
    the generator could widen its own evidence, Recall@K would stop describing what the
    answer was actually built from.

    The generator's contribution to honesty is narrow but concrete: it refuses when
    there is no evidence, and it reports citations that do not resolve rather than
    hiding them.
    """

    def __init__(
        self, model: LanguageModel, *, max_evidence: int = DEFAULT_MAX_EVIDENCE
    ) -> None:
        if max_evidence <= 0:
            raise ValueError(f"max_evidence must be positive, got {max_evidence}")

        self._model = model
        self._max_evidence = max_evidence

    def answer(self, query: str, results: Sequence[RetrievalResult]) -> GroundedAnswer:
        """Answer ``query`` from ``results``, or refuse if they cannot support one.

        With no results the model is never called. There is nothing to ground an answer
        in, and asking anyway would invite exactly the unsourced answer this pipeline
        exists to prevent — while costing a request to find that out.

        Raises:
            ValueError: the query is empty.
            GenerationError: the model returned nothing usable.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        if not results:
            return GroundedAnswer(
                query=query, text=NO_EVIDENCE_TEXT, evidence=(), has_answer=False
            )

        evidence = evidence_from_results(results[: self._max_evidence])
        text = self._model.complete(SYSTEM_PROMPT, build_prompt(query, evidence)).strip()
        if not text:
            raise GenerationError("the model returned an empty answer")

        if text == INSUFFICIENT_EVIDENCE:
            return GroundedAnswer(
                query=query, text=NO_EVIDENCE_TEXT, evidence=evidence, has_answer=False
            )

        citations, unresolved = _resolve_citations(text, evidence)
        return GroundedAnswer(
            query=query,
            text=text,
            evidence=evidence,
            citations=citations,
            unresolved_citations=unresolved,
        )


def _resolve_citations(
    text: str, evidence: Sequence[Evidence]
) -> tuple[tuple[Citation, ...], tuple[int, ...]]:
    """Map the markers cited in ``text`` back to the evidence they refer to.

    Markers are returned in order of first appearance and deduplicated, so a passage
    cited three times produces one source entry. A marker with no matching evidence is
    reported separately — silently dropping it would turn a hallucinated citation into
    an answer that merely looks lightly sourced.
    """
    by_marker = {item.marker: item for item in evidence}
    citations: list[Citation] = []
    unresolved: list[int] = []
    seen: set[int] = set()

    for match in _CITATION.finditer(text):
        for part in re.split(r"[,;]", match.group(1)):
            marker = int(part)
            if marker in seen:
                continue
            seen.add(marker)

            item = by_marker.get(marker)
            if item is None:
                unresolved.append(marker)
            else:
                citations.append(item.to_citation())

    return tuple(citations), tuple(unresolved)
