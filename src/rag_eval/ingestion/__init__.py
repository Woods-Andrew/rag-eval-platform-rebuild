"""PDF ingestion: files on disk become pages that know where they came from."""

from __future__ import annotations

from .errors import EmptyPDFError, PDFIngestionError, UnreadablePDFError
from .models import PDFPage
from .pdf_loader import load_pdf
from .text import clean_page_text

__all__ = [
    "EmptyPDFError",
    "PDFIngestionError",
    "PDFPage",
    "UnreadablePDFError",
    "clean_page_text",
    "load_pdf",
]
