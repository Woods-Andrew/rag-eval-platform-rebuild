"""Smoke tests: the package imports and the environment is the expected one."""

from __future__ import annotations

import sys

import rag_eval


def test_package_imports_with_version() -> None:
    assert rag_eval.__version__ == "0.1.0"


def test_running_on_python_311() -> None:
    """The project pins 3.11; a mismatched interpreter breaks reproducibility."""
    assert sys.version_info[:2] == (3, 11)


def test_pdf_backend_available() -> None:
    """PyMuPDF is the ingestion backend and must be importable offline."""
    import pymupdf

    assert pymupdf.__doc__ is not None
