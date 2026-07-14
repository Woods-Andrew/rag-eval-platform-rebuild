"""Tokenization for lexical retrieval."""

from __future__ import annotations

import re

__all__ = ["tokenize"]

# Alphanumeric runs, optionally joined by internal hyphens, underscores, or
# apostrophes: "multi-omics", "rag_eval", "don't", "bm25" are each one token.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_'][a-z0-9]+)*")

_JOINER = re.compile(r"[-_']")


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` into lexical tokens, emitting compounds and their parts.

    A hyphenated compound is emitted whole *and* split: "multi-omics" yields
    ``["multi-omics", "multi", "omics"]``. Without this, a query for "omics" scores
    zero against a document that only ever writes "multi-omics" — exactly the
    vocabulary-mismatch failure BM25 is otherwise good at avoiding. The compound is
    kept as well so an exact-phrase query still gets its strong, rare-term match.

    No stemming and no stopword list. BM25's IDF term already discounts words that
    appear everywhere, and stemming would blur the precise terminology that lexical
    search exists to catch.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        token = match.group()
        tokens.append(token)
        parts = _JOINER.split(token)
        if len(parts) > 1:
            tokens.extend(part for part in parts if part)
    return tokens
