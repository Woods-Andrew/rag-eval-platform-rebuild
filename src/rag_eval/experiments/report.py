"""Turning a sweep into something readable, and onto disk."""

from __future__ import annotations

import json
from pathlib import Path

from .sweep import SweepResult

__all__ = ["CROSS_VARIANT_CAVEAT", "format_sweep", "write_sweep"]

# Printed with every multi-variant sweep. Two chunking strategies are scored against
# two *different* label sets, because chunk IDs do not survive re-chunking — so a gap
# between variants mixes chunking quality with whatever labelling variance came with
# relabelling. Same-variant retriever comparisons share one label set and carry no such
# confound. Stating this next to the numbers is cheaper than explaining it afterwards.
CROSS_VARIANT_CAVEAT = (
    "note: variants are scored against separately labelled benchmarks, so differences "
    "between variants include labelling variance. Retriever comparisons within a "
    "variant share one label set and are directly comparable."
)


def format_sweep(sweep: SweepResult) -> str:
    """Render a sweep as plain text: corpus shapes first, then scores if there are any."""
    lines = [f"{sweep.document}: {len(sweep)} chunking variant(s), K={sweep.k}", ""]
    lines.extend(_corpus_table(sweep))

    scored = sweep.scored_variants
    if not scored:
        lines.extend(
            [
                "",
                "no variant had a benchmark, so only corpus shape was measured.",
                "retrieval quality is unmeasured — see data/evaluation/SCHEMA.md.",
            ]
        )
        return "\n".join(lines)

    lines.extend(["", "retrieval quality:", ""])
    lines.extend(_score_table(sweep))

    if sweep.is_comparable:
        lines.extend(["", CROSS_VARIANT_CAVEAT])

    unscored = [variant.name for variant in sweep.variants if not variant.was_scored]
    if unscored:
        lines.append(f"unscored (no benchmark): {', '.join(unscored)}")

    return "\n".join(lines)


def _corpus_table(sweep: SweepResult) -> list[str]:
    header = (
        f"{'variant':<14}{'chunks':>8}{'pages':>7}{'min':>7}{'median':>8}{'mean':>8}{'max':>7}"
    )
    rows = [header, "-" * len(header)]
    for variant in sweep.variants:
        stats = variant.stats
        rows.append(
            f"{variant.name:<14}"
            f"{stats.chunk_count:>8}"
            f"{stats.page_count:>7}"
            f"{stats.min_words:>7}"
            f"{stats.median_words:>8.1f}"
            f"{stats.mean_words:>8.1f}"
            f"{stats.max_words:>7}"
        )
    return rows


def _score_table(sweep: SweepResult) -> list[str]:
    header = f"{'variant':<14}{'retriever':<12}{'Recall@K':>10}{'nDCG@K':>10}{'misses':>9}"
    rows = [header, "-" * len(header)]
    for variant in sweep.scored_variants:
        for index, report in enumerate(variant.reports):
            rows.append(
                f"{variant.name if index == 0 else '':<14}"
                f"{report.retriever_name:<12}"
                f"{report.mean_recall:>10.4f}"
                f"{report.mean_ndcg:>10.4f}"
                f"{len(report.questions_with_no_hit):>9}"
            )
    return rows


def write_sweep(sweep: SweepResult, path: str | Path) -> Path:
    """Write the full sweep to JSON, per-question scores included.

    The per-question detail is the point: a headline mean is not auditable, but a list
    of which chunk IDs came back for which question can be checked against the document
    by hand. Parent directories are created so results can be written straight into a
    gitignored ``experiments/results/`` tree.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(sweep.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return destination
