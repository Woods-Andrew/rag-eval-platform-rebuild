"""Shared fixtures. PDFs are built on the fly so tests stay offline and deterministic."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[..., Path]:
    """Build a PDF from page texts and return its path.

    A page given as ``None`` is left blank, which is how a scanned or image-only
    page looks to a text extractor.
    """

    def _make_pdf(
        pages: Sequence[str | None] = ("page one",),
        *,
        name: str = "paper.pdf",
        metadata: dict[str, str] | None = None,
    ) -> Path:
        document = pymupdf.open()
        for text in pages:
            page = document.new_page()
            if text is not None:
                page.insert_text((72, 72), text.split("\n"))
        if metadata is not None:
            document.set_metadata(metadata)
        path = tmp_path / name
        document.save(path)
        document.close()
        return path

    return _make_pdf
