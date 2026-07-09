"""PDF file → :class:`PDFPage` objects, in document order."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from .errors import EmptyPDFError, UnreadablePDFError
from .models import PDFPage
from .text import clean_page_text

__all__ = ["load_pdf"]


def load_pdf(path: str | Path) -> list[PDFPage]:
    """Extract every page of a PDF, preserving source, page number, and metadata.

    Pages with no extractable text are still returned, so ``page_number`` stays
    aligned with the physical document. A PDF in which *no* page yields text is
    rejected instead: it is almost certainly scanned or image-only, and silently
    indexing zero text would produce a benchmark that measures nothing.

    Raises:
        FileNotFoundError: the path does not exist or is not a regular file.
        UnreadablePDFError: the file is not a parseable PDF, or is encrypted.
        EmptyPDFError: the PDF has no pages, or no page has extractable text.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"No PDF at {pdf_path}")

    try:
        document = pymupdf.open(pdf_path)
    except RuntimeError as exc:
        # PyMuPDF signals bad input with RuntimeError subclasses of its own
        # (FileDataError, FileNotFoundError) rather than the built-in exceptions.
        raise UnreadablePDFError(f"Could not open {pdf_path} as a PDF: {exc}") from exc

    with document:
        if document.needs_pass:
            raise UnreadablePDFError(f"{pdf_path.name} is password protected")
        if document.page_count == 0:
            raise EmptyPDFError(f"{pdf_path.name} contains no pages")

        metadata = _document_metadata(document)
        pages = [
            PDFPage(
                source=pdf_path.name,
                page_number=number,
                text=clean_page_text(page.get_text()),
                metadata=metadata,
            )
            for number, page in enumerate(document, start=1)
        ]

    if all(page.is_empty for page in pages):
        raise EmptyPDFError(
            f"{pdf_path.name} has {len(pages)} page(s) but no extractable text; "
            "it is likely a scanned or image-only PDF"
        )
    return pages


def _document_metadata(document: pymupdf.Document) -> dict[str, str]:
    """Document-level PDF metadata, dropping the keys the file left unset."""
    raw = document.metadata or {}
    return {key: value for key, value in raw.items() if isinstance(value, str) and value}
