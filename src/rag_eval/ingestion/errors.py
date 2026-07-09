"""Errors raised while turning a PDF into pages."""

from __future__ import annotations

__all__ = ["EmptyPDFError", "PDFIngestionError", "UnreadablePDFError"]


class PDFIngestionError(Exception):
    """Base class for ingestion failures.

    A missing file raises the built-in :class:`FileNotFoundError` instead, since
    that is what callers already expect from a path that does not exist.
    """


class UnreadablePDFError(PDFIngestionError):
    """The file exists but cannot be parsed as a PDF — corrupt, wrong format, or encrypted."""


class EmptyPDFError(PDFIngestionError):
    """The PDF parsed but yielded nothing to index — no pages, or no extractable text."""
