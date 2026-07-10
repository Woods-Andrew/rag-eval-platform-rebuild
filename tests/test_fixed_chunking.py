"""Fixed-size chunking: window mathematics, provenance, IDs, and validation."""

from __future__ import annotations

import pytest

from rag_eval.chunking import FixedSizeChunker, TextChunk, chunk_id_for
from rag_eval.ingestion import PDFPage


def page(text: str, *, number: int = 1, source: str = "paper.pdf", **metadata: str) -> PDFPage:
    return PDFPage(source=source, page_number=number, text=text, metadata=metadata)


def numbered_words(count: int) -> str:
    """``"w0 w1 w2 ..."`` — every word distinct, so windows are identifiable by content."""
    return " ".join(f"w{index}" for index in range(count))


class TestWindowing:
    def test_a_page_shorter_than_one_window_yields_a_single_chunk(self) -> None:
        chunks = FixedSizeChunker(chunk_size=10, overlap=2).chunk_page(page(numbered_words(4)))

        assert len(chunks) == 1
        assert chunks[0].text == numbered_words(4)

    def test_windows_advance_by_size_minus_overlap(self) -> None:
        chunks = FixedSizeChunker(chunk_size=10, overlap=4).chunk_page(page(numbered_words(25)))

        assert [chunk.text.split()[0] for chunk in chunks] == ["w0", "w6", "w12", "w18"]

    def test_consecutive_chunks_share_exactly_the_overlap(self) -> None:
        chunks = FixedSizeChunker(chunk_size=10, overlap=4).chunk_page(page(numbered_words(25)))

        first, second = chunks[0].text.split(), chunks[1].text.split()
        assert first[-4:] == second[:4]

    def test_zero_overlap_partitions_the_page_without_repetition(self) -> None:
        chunks = FixedSizeChunker(chunk_size=5, overlap=0).chunk_page(page(numbered_words(12)))

        words = [word for chunk in chunks for word in chunk.text.split()]
        assert words == numbered_words(12).split()

    def test_every_word_on_the_page_appears_in_some_chunk(self) -> None:
        chunks = FixedSizeChunker(chunk_size=7, overlap=3).chunk_page(page(numbered_words(30)))

        covered = {word for chunk in chunks for word in chunk.text.split()}
        assert covered == set(numbered_words(30).split())

    def test_the_final_short_window_is_not_a_duplicate_of_its_predecessor(self) -> None:
        # 20 words at size 10 / stride 5 lands the third window exactly on the end;
        # a naive loop would then emit a fourth window fully contained in the third.
        chunks = FixedSizeChunker(chunk_size=10, overlap=5).chunk_page(page(numbered_words(20)))

        assert len(chunks) == 3
        assert chunks[-1].text.split()[-1] == "w19"

    def test_windows_are_never_larger_than_the_configured_size(self) -> None:
        chunks = FixedSizeChunker(chunk_size=6, overlap=2).chunk_page(page(numbered_words(41)))

        assert all(chunk.word_count <= 6 for chunk in chunks)


class TestPageBoundaries:
    def test_chunks_never_span_two_pages(self) -> None:
        pages = [page(numbered_words(3), number=1), page("alpha beta gamma", number=2)]

        chunks = FixedSizeChunker(chunk_size=100, overlap=10).chunk_pages(pages)

        assert [chunk.page_number for chunk in chunks] == [1, 2]
        assert chunks[0].text == numbered_words(3)
        assert chunks[1].text == "alpha beta gamma"

    def test_blank_pages_produce_no_chunks_without_disturbing_numbering(self) -> None:
        pages = [page("alpha", number=1), page("   \n\n  ", number=2), page("gamma", number=3)]

        chunks = FixedSizeChunker().chunk_pages(pages)

        assert [chunk.page_number for chunk in chunks] == [1, 3]

    def test_an_entirely_empty_page_yields_an_empty_list(self) -> None:
        assert FixedSizeChunker().chunk_page(page("")) == []

    def test_chunk_index_restarts_on_each_page(self) -> None:
        pages = [page(numbered_words(12), number=n) for n in (1, 2)]

        chunks = FixedSizeChunker(chunk_size=5, overlap=0).chunk_pages(pages)

        assert [(c.page_number, c.chunk_index) for c in chunks] == [
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]


class TestProvenance:
    def test_source_page_and_metadata_survive_chunking(self) -> None:
        source_page = page(numbered_words(30), number=7, source="omics.pdf", title="Multi-Omics")

        chunks = FixedSizeChunker(chunk_size=10, overlap=2).chunk_page(source_page)

        assert all(chunk.source == "omics.pdf" for chunk in chunks)
        assert all(chunk.page_number == 7 for chunk in chunks)
        assert all(chunk.metadata["title"] == "Multi-Omics" for chunk in chunks)

    def test_chunk_metadata_is_read_only(self) -> None:
        chunk = FixedSizeChunker().chunk_page(page("alpha", title="Multi-Omics"))[0]

        with pytest.raises(TypeError):
            chunk.metadata["title"] = "rewritten"  # type: ignore[index]

    def test_citation_reads_as_source_and_page(self) -> None:
        chunk = FixedSizeChunker().chunk_page(page("alpha", number=3, source="omics.pdf"))[0]

        assert chunk.citation == "omics.pdf p.3"

    def test_line_structure_inside_a_chunk_is_preserved_verbatim(self) -> None:
        chunks = FixedSizeChunker(chunk_size=10, overlap=0).chunk_page(page("alpha\n\nbeta gamma"))

        assert chunks[0].text == "alpha\n\nbeta gamma"


class TestChunkIDs:
    def test_ids_are_stable_across_runs(self) -> None:
        chunker = FixedSizeChunker(chunk_size=8, overlap=3)
        source_page = page(numbered_words(40), number=2)

        first = [chunk.chunk_id for chunk in chunker.chunk_page(source_page)]
        second = [chunk.chunk_id for chunk in chunker.chunk_page(source_page)]

        assert first == second

    def test_ids_are_unique_within_a_document(self) -> None:
        pages = [page(numbered_words(50), number=n) for n in (1, 2, 3)]

        chunks = FixedSizeChunker(chunk_size=9, overlap=4).chunk_pages(pages)

        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    def test_identical_text_on_different_pages_gets_different_ids(self) -> None:
        pages = [page("shared boilerplate", number=1), page("shared boilerplate", number=2)]

        chunks = FixedSizeChunker().chunk_pages(pages)

        assert chunks[0].chunk_id != chunks[1].chunk_id

    def test_changing_the_text_changes_the_id(self) -> None:
        original = chunk_id_for("paper.pdf", 1, 0, "the imputation strategy")
        edited = chunk_id_for("paper.pdf", 1, 0, "the imputation strategies")

        assert original != edited

    def test_id_carries_readable_provenance(self) -> None:
        chunk = FixedSizeChunker().chunk_page(page("alpha", number=12, source="omics.pdf"))[0]

        assert chunk.chunk_id.startswith("omics-p012-c00-")


class TestChunkerValidation:
    @pytest.mark.parametrize("size", [0, -1])
    def test_chunk_size_must_be_positive(self, size: int) -> None:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            FixedSizeChunker(chunk_size=size)

    def test_overlap_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="overlap must not be negative"):
            FixedSizeChunker(chunk_size=10, overlap=-1)

    @pytest.mark.parametrize("overlap", [10, 11])
    def test_overlap_below_chunk_size_is_required(self, overlap: int) -> None:
        # Equal size and overlap gives a stride of zero: the window would never advance.
        with pytest.raises(ValueError, match="never advances"):
            FixedSizeChunker(chunk_size=10, overlap=overlap)

    def test_stride_is_size_minus_overlap(self) -> None:
        assert FixedSizeChunker(chunk_size=200, overlap=40).stride == 160


class TestTextChunkModel:
    def test_page_number_must_be_one_indexed(self) -> None:
        with pytest.raises(ValueError, match="1-indexed"):
            TextChunk(chunk_id="x", text="alpha", source="p.pdf", page_number=0, chunk_index=0)

    def test_chunk_index_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="0-based"):
            TextChunk(chunk_id="x", text="alpha", source="p.pdf", page_number=1, chunk_index=-1)

    def test_source_must_be_present(self) -> None:
        with pytest.raises(ValueError, match="non-empty filename"):
            TextChunk(chunk_id="x", text="alpha", source="", page_number=1, chunk_index=0)

    def test_an_empty_chunk_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must contain text"):
            TextChunk(chunk_id="x", text="   \n ", source="p.pdf", page_number=1, chunk_index=0)

    def test_chunks_are_immutable(self) -> None:
        chunk = TextChunk(chunk_id="x", text="alpha", source="p.pdf", page_number=1, chunk_index=0)

        with pytest.raises(AttributeError):
            chunk.text = "rewritten"  # type: ignore[misc]

    def test_word_count_counts_whitespace_separated_words(self) -> None:
        chunk = TextChunk(
            chunk_id="x",
            text="alpha  beta\ngamma",
            source="p.pdf",
            page_number=1,
            chunk_index=0,
        )

        assert chunk.word_count == 3
