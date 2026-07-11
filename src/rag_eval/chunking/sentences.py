"""Approximate sentence segmentation, tuned for academic prose."""

from __future__ import annotations

import re

__all__ = ["split_sentences"]

# Lowercase, without the trailing period. A period following one of these is part of
# the abbreviation, not the end of a sentence — "et al. (2024) showed" is one sentence.
_ABBREVIATIONS = frozenset(
    {
        "al",
        "approx",
        "cf",
        "ch",
        "dr",
        "e.g",
        "eq",
        "eqs",
        "etc",
        "et al",
        "fig",
        "figs",
        "i.e",
        "inc",
        "mr",
        "ms",
        "no",
        "nos",
        "pp",
        "prof",
        "ref",
        "refs",
        "sec",
        "st",
        "tab",
        "vol",
        "vs",
    }
)

# A boundary is terminal punctuation, optional closing quotes or brackets, then
# whitespace. Requiring the whitespace is what makes decimals ("0.95") and dotted
# identifiers ("rag_eval.chunking") safe without any special handling.
_BOUNDARY = re.compile(r"""(?<=[.!?])["')\]]*\s+""")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, keeping their internal whitespace intact.

    This is a heuristic, not a parser: it knows about a fixed list of academic
    abbreviations and about initials, and it gets decimals right by construction. It
    will still mis-split unusual constructions. That is acceptable here because the
    only consumer is chunking, where a wrong boundary costs a slightly awkward chunk
    edge rather than a wrong answer.
    """
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if not _is_sentence_end(text, match.start()):
            continue
        sentence = text[start : match.start()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _is_sentence_end(text: str, index: int) -> bool:
    """True when the punctuation ending ``text[:index]`` really closes a sentence."""
    before = text[:index].rstrip("\"')]")
    if not before.endswith("."):
        return True  # '!' and '?' do not appear inside abbreviations

    tokens = before[:-1].split()
    if not tokens:
        return True

    token = tokens[-1].lstrip("([\"'")
    if token.lower() in _ABBREVIATIONS:
        return False
    # An initial in a name: "J. Smith", "A. B. Author".
    return not (len(token) == 1 and token.isalpha())
