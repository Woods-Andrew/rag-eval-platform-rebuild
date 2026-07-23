"""Parsing a chunking variant out of a one-line command line spec."""

from __future__ import annotations

from pathlib import Path

from ..chunking import Chunker, FixedSizeChunker, StructureAwareChunker
from .sweep import ChunkingVariant

__all__ = ["SPEC_HELP", "parse_variant"]

SPEC_HELP = (
    "chunking variant as STRATEGY[:A/B][=BENCHMARK.json]; "
    "fixed:SIZE/OVERLAP or structure:MAX/MIN, e.g. fixed:300/60=labels.json"
)


def parse_variant(spec: str) -> ChunkingVariant:
    """Build a ``ChunkingVariant`` from ``strategy[:a/b][=benchmark]``.

    Parameters are part of the spec because chunk size is the variable a chunking
    experiment is actually sweeping — comparing only the two default configurations
    would answer a much narrower question than "what size works for this document".

    The variant is named after its own spec, so a results table labels each row with
    the configuration that produced it and two rows can never silently collide.

    Raises:
        ValueError: unknown strategy, malformed parameters, or a benchmark path that
            does not exist.
    """
    chunker_spec, separator, benchmark_spec = spec.partition("=")
    chunker_spec = chunker_spec.strip()
    if not chunker_spec:
        raise ValueError(f"empty chunking variant in {spec!r}")

    benchmark_path: Path | None = None
    if separator:
        if not benchmark_spec.strip():
            raise ValueError(f"{spec!r} has '=' but no benchmark path")
        benchmark_path = Path(benchmark_spec.strip())
        if not benchmark_path.is_file():
            raise ValueError(f"no benchmark file at {benchmark_path}")

    strategy, _, parameters = chunker_spec.partition(":")
    return ChunkingVariant(
        name=chunker_spec,
        chunker=_build_chunker(strategy.strip().lower(), parameters.strip(), spec),
        benchmark_path=benchmark_path,
    )


def _build_chunker(strategy: str, parameters: str, spec: str) -> Chunker:
    if strategy == "fixed":
        if not parameters:
            return FixedSizeChunker()
        size, overlap = _parse_pair(parameters, spec, names=("SIZE", "OVERLAP"))
        return FixedSizeChunker(chunk_size=size, overlap=overlap)

    if strategy == "structure":
        if not parameters:
            return StructureAwareChunker()
        maximum, minimum = _parse_pair(parameters, spec, names=("MAX", "MIN"))
        return StructureAwareChunker(max_words=maximum, min_words=minimum)

    raise ValueError(f"unknown chunking strategy {strategy!r} in {spec!r}; expected fixed or structure")


def _parse_pair(parameters: str, spec: str, *, names: tuple[str, str]) -> tuple[int, int]:
    first, separator, second = parameters.partition("/")
    if not separator:
        raise ValueError(f"{spec!r} needs both parameters as {names[0]}/{names[1]}")
    try:
        return int(first), int(second)
    except ValueError:
        raise ValueError(
            f"{spec!r} has non-integer parameters; expected {names[0]}/{names[1]}"
        ) from None
