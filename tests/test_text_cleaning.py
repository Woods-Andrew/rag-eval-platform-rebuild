"""Text normalization applied between extraction and chunking."""

from __future__ import annotations

from rag_eval.ingestion import clean_page_text


def test_ligatures_and_nbsp_normalize_to_ascii() -> None:
    """NFKC is what makes "ﬁne" findable by a search for "fine"."""
    assert clean_page_text("the ﬁrst ﬂag here") == "the first flag here"


def test_hyphenated_line_break_is_rejoined() -> None:
    assert clean_page_text("informa-\ntion retrieval") == "information retrieval"


def test_hyphen_not_at_a_line_break_is_left_alone() -> None:
    assert clean_page_text("multi-omics embedding") == "multi-omics embedding"


def test_paragraph_breaks_survive_but_runs_collapse() -> None:
    """Structure-aware chunking reads blank lines, so they must not be flattened."""
    assert clean_page_text("first para\n\n\n\nsecond para") == "first para\n\nsecond para"


def test_horizontal_whitespace_runs_collapse() -> None:
    assert clean_page_text("column   one\t\tcolumn two") == "column one column two"


def test_carriage_returns_and_control_characters_are_removed() -> None:
    assert clean_page_text("line one\r\nline\x00 two\r") == "line one\nline two"


def test_surrounding_whitespace_is_stripped() -> None:
    assert clean_page_text("\n\n  padded text  \n\n") == "padded text"


def test_empty_input_stays_empty() -> None:
    assert clean_page_text("") == ""
    assert clean_page_text("   \n\n  ") == ""
