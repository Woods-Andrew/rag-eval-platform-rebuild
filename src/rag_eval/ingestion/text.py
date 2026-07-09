"""Normalization applied to raw PDF text before it reaches chunking."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["clean_page_text"]

# Control characters that carry no meaning in extracted text. Tab and newline are
# deliberately excluded — they are handled as whitespace below.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_HORIZONTAL_RUN = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_RUN = re.compile(r"\n{3,}")


def clean_page_text(raw: str) -> str:
    """Normalize text extracted from a PDF page without changing its wording.

    Applies Unicode NFKC so typographic ligatures ("ﬁ") and non-breaking spaces
    become their searchable ASCII equivalents, rejoins words split across a line
    break by hyphenation, and collapses the whitespace noise PDF extraction
    produces. Blank lines are preserved as paragraph boundaries because
    structure-aware chunking relies on them.

    Rejoining hyphens is the right call for soft hyphens ("informa-\\ntion"), but
    it also fuses a genuine compound that happens to break at a line end
    ("multi-\\nomics" becomes "multiomics"). Both readings lose information for
    some query; this picks the one that is correct far more often.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _HORIZONTAL_RUN.sub(" ", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()
