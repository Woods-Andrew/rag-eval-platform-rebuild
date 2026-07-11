"""Structure-aware chunking: heading detection, the split ladder, and provenance."""

from __future__ import annotations

import pytest

from rag_eval.chunking import Chunker, FixedSizeChunker, StructureAwareChunker, is_heading
from rag_eval.ingestion import PDFPage


def page(text: str, *, number: int = 1, source: str = "omics.pdf", **metadata: str) -> PDFPage:
    return PDFPage(source=source, page_number=number, text=text, metadata=metadata)


def sentences(count: int, *, words: int = 10) -> str:
    """``count`` sentences of exactly ``words`` words each, every one distinct."""
    return " ".join(
        " ".join([f"s{index}"] + ["filler"] * (words - 2) + ["end."]) for index in range(count)
    )


class TestHeadingDetection:
    @pytest.mark.parametrize(
        "line",
        [
            "Abstract",
            "Introduction",
            "Related Work",
            "Materials and Methods",
            "References",
            "Results:",
        ],
    )
    def test_known_section_names_are_headings(self, line: str) -> None:
        assert is_heading(line)

    @pytest.mark.parametrize(
        "line",
        ["3 Methods", "3.2 Imputation Strategy", "3.2.1 Learned Priors", "4. Results"],
    )
    def test_numbered_headings_are_detected(self, line: str) -> None:
        assert is_heading(line)

    def test_all_caps_lines_are_headings(self) -> None:
        assert is_heading("EXPERIMENTAL SETUP")

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "   ",
            "We introduce a hybrid retriever.",
            "The results, however, were mixed",
            "This paragraph is far too long to plausibly be a section heading in any "
            "technical document whatsoever",
        ],
    )
    def test_prose_is_not_a_heading(self, line: str) -> None:
        assert not is_heading(line)

    def test_title_case_alone_is_not_enough(self) -> None:
        # Deliberately not detected: this pattern fires on captions and author lines.
        assert not is_heading("A Hybrid Retriever For Technical Documents")


class TestSectionBoundaries:
    def test_a_heading_starts_a_new_chunk(self) -> None:
        text = "Introduction\n\nalpha beta gamma\n\nMethods\n\ndelta epsilon"

        chunks = StructureAwareChunker(max_words=100, min_words=1).chunk_page(page(text))

        assert [chunk.section for chunk in chunks] == ["Introduction", "Methods"]

    def test_chunks_are_tagged_with_the_heading_above_them(self) -> None:
        text = "2 Methods\n\n" + sentences(4) + "\n\n" + sentences(4)

        chunks = StructureAwareChunker(max_words=25, min_words=1).chunk_page(page(text))

        assert len(chunks) > 1
        assert all(chunk.section == "2 Methods" for chunk in chunks)

    def test_text_before_any_heading_has_no_section(self) -> None:
        text = "orphan text with no heading\n\nIntroduction\n\nalpha beta"

        chunks = StructureAwareChunker(max_words=100, min_words=1).chunk_page(page(text))

        assert chunks[0].section is None
        assert chunks[1].section == "Introduction"

    def test_sections_do_not_carry_across_pages(self) -> None:
        pages = [page("Methods\n\nalpha beta", number=1), page("gamma delta", number=2)]

        chunks = StructureAwareChunker(max_words=100, min_words=1).chunk_pages(pages)

        assert chunks[0].section == "Methods"
        assert chunks[1].section is None

    def test_the_heading_is_repeated_in_each_chunk_body(self) -> None:
        text = "2 Methods\n\n" + sentences(6)

        chunks = StructureAwareChunker(max_words=25, min_words=1).chunk_page(page(text))

        assert len(chunks) > 1
        assert all(chunk.text.startswith("2 Methods") for chunk in chunks)

    def test_heading_repetition_can_be_switched_off(self) -> None:
        text = "2 Methods\n\nalpha beta gamma"

        chunker = StructureAwareChunker(max_words=100, min_words=1, include_heading=False)
        chunk = chunker.chunk_page(page(text))[0]

        assert chunk.text == "alpha beta gamma"
        assert chunk.section == "2 Methods"


class TestSplitLadder:
    def test_paragraphs_accumulate_up_to_the_limit(self) -> None:
        text = "alpha beta\n\ngamma delta\n\nepsilon zeta"

        chunks = StructureAwareChunker(max_words=100, min_words=1).chunk_page(page(text))

        assert len(chunks) == 1
        assert chunks[0].text == "alpha beta\n\ngamma delta\n\nepsilon zeta"

    def test_a_paragraph_that_would_overflow_starts_a_new_chunk(self) -> None:
        text = sentences(2) + "\n\n" + sentences(2)

        chunks = StructureAwareChunker(max_words=25, min_words=1).chunk_page(page(text))

        assert len(chunks) == 2

    def test_an_oversized_paragraph_is_split_between_sentences(self) -> None:
        chunks = StructureAwareChunker(max_words=25, min_words=1).chunk_page(page(sentences(6)))

        assert len(chunks) > 1
        assert all(chunk.text.rstrip().endswith("end.") for chunk in chunks)

    def test_a_single_oversized_sentence_falls_back_to_word_windows(self) -> None:
        one_long_sentence = "alpha " * 59 + "omega."

        chunks = StructureAwareChunker(max_words=25, min_words=1).chunk_page(
            page(one_long_sentence)
        )

        assert [chunk.word_count for chunk in chunks] == [25, 25, 10]

    def test_no_chunk_exceeds_the_limit_when_no_tail_is_merged(self) -> None:
        chunks = StructureAwareChunker(max_words=30, min_words=1).chunk_page(page(sentences(20)))

        assert all(chunk.word_count <= 30 for chunk in chunks)


class TestShortTailMerging:
    def test_a_runt_final_chunk_is_folded_into_its_predecessor(self) -> None:
        # Six 10-word sentences at a 25-word limit leaves a 20/20/20 split; dropping
        # to five leaves a 20/20/10 split whose tail is under min_words.
        text = sentences(5)

        merged = StructureAwareChunker(max_words=25, min_words=15).chunk_page(page(text))
        unmerged = StructureAwareChunker(max_words=25, min_words=1).chunk_page(page(text))

        assert len(merged) == len(unmerged) - 1
        assert merged[-1].word_count > 25

    def test_merging_only_happens_within_one_section(self) -> None:
        text = "Introduction\n\nalpha beta\n\nMethods\n\ngamma delta"

        chunks = StructureAwareChunker(max_words=100, min_words=50).chunk_page(page(text))

        assert [chunk.section for chunk in chunks] == ["Introduction", "Methods"]

    def test_a_lone_short_chunk_is_still_emitted(self) -> None:
        chunks = StructureAwareChunker(max_words=100, min_words=50).chunk_page(page("alpha beta"))

        assert len(chunks) == 1
        assert chunks[0].text == "alpha beta"


class TestProvenanceAndCoverage:
    def test_every_word_of_the_page_survives_chunking(self) -> None:
        text = "2 Methods\n\n" + sentences(8)
        chunker = StructureAwareChunker(max_words=25, min_words=1)

        covered = {word for chunk in chunker.chunk_page(page(text)) for word in chunk.text.split()}

        assert covered == set(text.split())

    def test_without_heading_repetition_the_heading_lives_only_in_section(self) -> None:
        # The heading is consumed as a structural marker, so with include_heading off
        # it is absent from every chunk body. Provenance is kept on `section`.
        text = "2 Methods\n\n" + sentences(8)
        chunker = StructureAwareChunker(max_words=25, min_words=1, include_heading=False)

        chunks = chunker.chunk_page(page(text))
        covered = {word for chunk in chunks for word in chunk.text.split()}

        assert covered == set(sentences(8).split())
        assert all(chunk.section == "2 Methods" for chunk in chunks)

    def test_source_page_and_metadata_survive(self) -> None:
        source_page = page(sentences(8), number=5, source="omics.pdf", title="Multi-Omics")

        chunks = StructureAwareChunker(max_words=25, min_words=1).chunk_page(source_page)

        assert all(chunk.source == "omics.pdf" for chunk in chunks)
        assert all(chunk.page_number == 5 for chunk in chunks)
        assert all(chunk.metadata["title"] == "Multi-Omics" for chunk in chunks)

    def test_citation_includes_the_section(self) -> None:
        chunk = StructureAwareChunker(min_words=1).chunk_page(
            page("Methods\n\nalpha beta", number=3)
        )[0]

        assert chunk.citation == "omics.pdf p.3 § Methods"

    def test_chunk_ids_are_unique_and_stable(self) -> None:
        chunker = StructureAwareChunker(max_words=25, min_words=1)
        source_page = page(sentences(12), number=2)

        first = [chunk.chunk_id for chunk in chunker.chunk_page(source_page)]
        second = [chunk.chunk_id for chunk in chunker.chunk_page(source_page)]

        assert first == second
        assert len(set(first)) == len(first)

    def test_chunk_index_restarts_on_each_page(self) -> None:
        pages = [page("alpha beta", number=n) for n in (1, 2)]

        chunks = StructureAwareChunker(min_words=1).chunk_pages(pages)

        assert [(c.page_number, c.chunk_index) for c in chunks] == [(1, 0), (2, 0)]

    def test_blank_pages_produce_no_chunks(self) -> None:
        assert StructureAwareChunker().chunk_page(page("   \n\n  ")) == []


class TestValidation:
    def test_max_words_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_words must be positive"):
            StructureAwareChunker(max_words=0)

    def test_min_words_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="min_words must not be negative"):
            StructureAwareChunker(min_words=-1)

    def test_min_words_must_be_below_max_words(self) -> None:
        with pytest.raises(ValueError, match="must be smaller than max_words"):
            StructureAwareChunker(max_words=50, min_words=50)


class TestChunkerProtocol:
    @pytest.mark.parametrize(
        "chunker", [FixedSizeChunker(), StructureAwareChunker()], ids=["fixed", "structure"]
    )
    def test_both_strategies_satisfy_the_protocol(self, chunker: Chunker) -> None:
        assert isinstance(chunker, Chunker)

    def test_both_strategies_are_interchangeable(self) -> None:
        pages = [page(sentences(8), number=1)]

        for chunker in (FixedSizeChunker(chunk_size=25, overlap=0), StructureAwareChunker()):
            chunks = chunker.chunk_pages(pages)
            assert chunks and all(chunk.source == "omics.pdf" for chunk in chunks)
