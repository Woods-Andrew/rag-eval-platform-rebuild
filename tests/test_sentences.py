"""Sentence segmentation: the boundaries academic prose makes ambiguous."""

from __future__ import annotations

import pytest

from rag_eval.chunking import split_sentences


class TestPlainBoundaries:
    def test_splits_on_terminal_punctuation(self) -> None:
        assert split_sentences("First one. Second one! Third one?") == [
            "First one.",
            "Second one!",
            "Third one?",
        ]

    def test_text_without_terminal_punctuation_is_one_sentence(self) -> None:
        assert split_sentences("a heading with no period") == ["a heading with no period"]

    def test_empty_and_blank_input_yield_nothing(self) -> None:
        assert split_sentences("") == []
        assert split_sentences("   \n\n  ") == []

    def test_a_boundary_across_a_line_break_still_splits(self) -> None:
        assert split_sentences("First one.\nSecond one.") == ["First one.", "Second one."]

    def test_closing_quotes_and_brackets_stay_with_their_sentence(self) -> None:
        assert split_sentences('He said "no." Then he left.') == [
            'He said "no."',
            "Then he left.",
        ]


class TestAmbiguousPeriods:
    def test_decimals_are_not_boundaries(self) -> None:
        # No whitespace after the period, so the regex never even proposes a break.
        assert split_sentences("Recall@5 was 0.95 overall.") == ["Recall@5 was 0.95 overall."]

    @pytest.mark.parametrize(
        "text",
        [
            "As Chen et al. (2024) showed, it helps.",
            "See Fig. 3 for the ablation.",
            "Results appear in Sec. 4 below.",
            "Compare with e.g. the dense baseline.",
            "That is, i.e. the fused ranking.",
        ],
    )
    def test_abbreviations_do_not_end_a_sentence(self, text: str) -> None:
        assert split_sentences(text) == [text]

    def test_name_initials_do_not_end_a_sentence(self) -> None:
        assert split_sentences("Written by J. Smith and A. B. Doe. The results follow.") == [
            "Written by J. Smith and A. B. Doe.",
            "The results follow.",
        ]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # "no." (number) and "ms." (honorific) are kept out of the abbreviation
            # list on purpose: they collide with the ordinary word and with the unit.
            ("The answer is no. We moved on.", ["The answer is no.", "We moved on."]),
            ("Latency was 40 ms. That is fast.", ["Latency was 40 ms.", "That is fast."]),
        ],
    )
    def test_common_words_are_not_treated_as_abbreviations(
        self, text: str, expected: list[str]
    ) -> None:
        assert split_sentences(text) == expected

    def test_an_abbreviation_mid_sentence_still_splits_at_the_real_end(self) -> None:
        assert split_sentences("See Fig. 3 for details. It plots nDCG.") == [
            "See Fig. 3 for details.",
            "It plots nDCG.",
        ]


class TestContentPreservation:
    def test_no_words_are_lost(self) -> None:
        text = "First one. Second one! Third one? A trailing fragment"

        rejoined = " ".join(split_sentences(text)).split()

        assert rejoined == text.split()

    def test_closing_brackets_are_not_dropped(self) -> None:
        # The closers are captured with the punctuation rather than skipped as part
        # of the separator, so they stay attached instead of vanishing.
        assert split_sentences("It works (mostly.) Then it broke.") == [
            "It works (mostly.)",
            "Then it broke.",
        ]
