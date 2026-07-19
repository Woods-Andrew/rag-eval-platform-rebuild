"""Loading a hand-labelled benchmark, and refusing to load a broken one."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..retrieval import Corpus
from .models import BenchmarkQuestion

__all__ = ["Benchmark", "BenchmarkError", "load_benchmark"]


class BenchmarkError(ValueError):
    """The benchmark file is malformed, or disagrees with the corpus it labels."""


@dataclass(frozen=True)
class Benchmark:
    """A set of labelled questions, plus the provenance of how they were labelled."""

    questions: tuple[BenchmarkQuestion, ...]
    document: str
    chunking: str
    notes: str = ""

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Any:
        return iter(self.questions)

    @property
    def categories(self) -> tuple[str, ...]:
        """Distinct question categories, sorted."""
        return tuple(sorted({question.category for question in self.questions}))

    @property
    def label_count(self) -> int:
        return sum(len(question.relevant_chunk_ids) for question in self.questions)


def load_benchmark(path: str | Path, corpus: Corpus | None = None) -> Benchmark:
    """Read a benchmark file, validating it against ``corpus`` when one is given.

    The corpus check is the reason this function exists rather than a bare
    ``json.load``. Relevance labels reference chunks by ID, and a chunk ID encodes both
    its provenance and a digest of its text — so a label that no longer resolves means
    the document or the chunking settings changed underneath it. Scoring against that
    would silently measure the wrong thing, so it is an error, not a warning.

    Raises:
        FileNotFoundError: no benchmark file at ``path``.
        BenchmarkError: malformed JSON, a missing field, a duplicate question ID, or a
            label that does not match any chunk in ``corpus``.
    """
    benchmark_path = Path(path)
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"No benchmark file at {benchmark_path}")

    try:
        raw = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{benchmark_path.name} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise BenchmarkError(f"{benchmark_path.name} must contain a JSON object at the top level")

    questions = _parse_questions(raw.get("questions"), benchmark_path.name)
    if corpus is not None:
        _validate_against_corpus(questions, corpus, benchmark_path.name)

    return Benchmark(
        questions=questions,
        document=str(raw.get("document", "")),
        chunking=str(raw.get("chunking", "")),
        notes=str(raw.get("notes", "")),
    )


def _parse_questions(raw: object, filename: str) -> tuple[BenchmarkQuestion, ...]:
    if not isinstance(raw, list) or not raw:
        raise BenchmarkError(f"{filename} must contain a non-empty 'questions' array")

    questions: list[BenchmarkQuestion] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise BenchmarkError(f"{filename}: question {index} is not an object")

        question_id = str(entry.get("id", "")).strip()
        if not question_id:
            raise BenchmarkError(f"{filename}: question {index} has no 'id'")
        if question_id in seen:
            raise BenchmarkError(f"{filename}: duplicate question id {question_id!r}")
        seen.add(question_id)

        relevant = entry.get("relevant_chunk_ids")
        if not isinstance(relevant, list) or not all(isinstance(item, str) for item in relevant):
            raise BenchmarkError(
                f"{filename}: {question_id} needs 'relevant_chunk_ids' as a list of strings"
            )

        try:
            questions.append(
                BenchmarkQuestion(
                    question_id=question_id,
                    query=str(entry.get("query", "")),
                    relevant_chunk_ids=frozenset(relevant),
                    category=str(entry.get("category", "uncategorized")),
                    notes=str(entry.get("notes", "")),
                )
            )
        except ValueError as exc:
            raise BenchmarkError(f"{filename}: {exc}") from exc

    return tuple(questions)


def _validate_against_corpus(
    questions: Sequence[BenchmarkQuestion], corpus: Corpus, filename: str
) -> None:
    dangling = {
        question.question_id: sorted(
            chunk_id for chunk_id in question.relevant_chunk_ids if chunk_id not in corpus
        )
        for question in questions
    }
    broken = {question_id: ids for question_id, ids in dangling.items() if ids}
    if broken:
        detail = "; ".join(
            f"{question_id} -> {', '.join(ids)}" for question_id, ids in sorted(broken.items())
        )
        raise BenchmarkError(
            f"{filename}: {len(broken)} question(s) reference chunk IDs that are not in the "
            f"corpus, so the document or the chunking settings changed since labelling "
            f"({detail})"
        )
