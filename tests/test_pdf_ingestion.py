"""PDF ingestion: page extraction, provenance, and rejection of unusable input."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rag_eval.ingestion import (
    EmptyPDFError,
    PDFIngestionError,
    PDFPage,
    UnreadablePDFError,
    load_pdf,
)

# A hand-written PDF with an empty page tree. PyMuPDF refuses to *save* a document
# with zero pages, so the only way to exercise that branch is to write the bytes.
ZERO_PAGE_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj
trailer<</Root 1 0 R>>
"""

PdfFactory = Callable[..., Path]


class TestPageExtraction:
    def test_every_page_is_returned_in_document_order(self, make_pdf: PdfFactory) -> None:
        pages = load_pdf(make_pdf(["alpha", "beta", "gamma"]))

        assert [page.text for page in pages] == ["alpha", "beta", "gamma"]

    def test_page_numbers_are_one_indexed(self, make_pdf: PdfFactory) -> None:
        pages = load_pdf(make_pdf(["alpha", "beta", "gamma"]))

        assert [page.page_number for page in pages] == [1, 2, 3]

    def test_source_is_the_filename(self, make_pdf: PdfFactory) -> None:
        pages = load_pdf(make_pdf(["alpha"], name="agentic-omics.pdf"))

        assert all(page.source == "agentic-omics.pdf" for page in pages)

    def test_extracted_text_is_cleaned(self, make_pdf: PdfFactory) -> None:
        """The loader hands chunking normalized text, not raw extractor output."""
        pages = load_pdf(make_pdf(["spaced    out\ntext   here"]))

        assert pages[0].text == "spaced out\ntext here"

    def test_blank_pages_are_kept_so_numbering_stays_aligned(
        self, make_pdf: PdfFactory
    ) -> None:
        pages = load_pdf(make_pdf(["alpha", None, "gamma"]))

        assert len(pages) == 3
        assert pages[1].text == ""
        assert pages[1].is_empty
        assert pages[2].page_number == 3


class TestMetadata:
    def test_document_metadata_is_attached_to_every_page(
        self, make_pdf: PdfFactory
    ) -> None:
        path = make_pdf(
            ["alpha", "beta"],
            metadata={"title": "Agentic AI for Multi-Omics", "author": "A. Woods"},
        )

        pages = load_pdf(path)

        for page in pages:
            assert page.metadata["title"] == "Agentic AI for Multi-Omics"
            assert page.metadata["author"] == "A. Woods"

    def test_unset_metadata_keys_are_dropped(self, make_pdf: PdfFactory) -> None:
        """PyMuPDF reports every key with an empty value; empty provenance is noise."""
        pages = load_pdf(make_pdf(["alpha"], metadata={"title": "Only A Title"}))

        assert pages[0].metadata["title"] == "Only A Title"
        assert "author" not in pages[0].metadata
        assert all(value for value in pages[0].metadata.values())

    def test_metadata_is_read_only(self, make_pdf: PdfFactory) -> None:
        pages = load_pdf(make_pdf(["alpha"], metadata={"title": "T"}))

        with pytest.raises(TypeError):
            pages[0].metadata["title"] = "tampered"  # type: ignore[index]


class TestInvalidInput:
    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path / "does-not-exist.pdf")

    def test_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_pdf(tmp_path)

    def test_corrupt_file_raises_unreadable(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.pdf"
        path.write_bytes(b"this is not a PDF, it is just bytes")

        with pytest.raises(UnreadablePDFError):
            load_pdf(path)

    def test_pdf_without_pages_raises_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "zero-pages.pdf"
        path.write_bytes(ZERO_PAGE_PDF)

        with pytest.raises(EmptyPDFError, match="no pages"):
            load_pdf(path)

    def test_pdf_without_extractable_text_raises_empty(
        self, make_pdf: PdfFactory
    ) -> None:
        """A scanned paper must fail loudly rather than index an empty corpus."""
        with pytest.raises(EmptyPDFError, match="image-only"):
            load_pdf(make_pdf([None, None]))

    def test_ingestion_errors_share_a_base_class(self) -> None:
        """Callers can catch one exception type for any unusable document."""
        assert issubclass(UnreadablePDFError, PDFIngestionError)
        assert issubclass(EmptyPDFError, PDFIngestionError)


class TestPDFPageModel:
    def test_page_numbers_below_one_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-indexed"):
            PDFPage(source="paper.pdf", page_number=0, text="text")

    def test_source_is_required(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            PDFPage(source="", page_number=1, text="text")

    def test_page_is_frozen(self) -> None:
        page = PDFPage(source="paper.pdf", page_number=1, text="text")

        with pytest.raises(AttributeError):
            page.text = "tampered"  # type: ignore[misc]

    def test_metadata_defaults_to_empty(self) -> None:
        assert PDFPage(source="paper.pdf", page_number=1, text="text").metadata == {}

    def test_metadata_is_copied_from_the_caller(self) -> None:
        """Mutating the source dict afterwards must not rewrite a page's provenance."""
        supplied = {"title": "Original"}
        page = PDFPage(source="paper.pdf", page_number=1, text="text", metadata=supplied)

        supplied["title"] = "Changed"

        assert page.metadata["title"] == "Original"
