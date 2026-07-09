"""Data model for an ingested PDF page."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = ["PDFPage"]


@dataclass(frozen=True)
class PDFPage:
    """One page of a PDF, carrying the provenance needed to cite it later.

    ``page_number`` is 1-indexed so it matches the page a reader would turn to.
    ``metadata`` is document-level (title, author, ...) and is therefore identical
    across every page of the same document; it is copied into a read-only mapping
    so a page cannot be mutated after ingestion.
    """

    source: str
    page_number: int
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must be a non-empty filename")
        if self.page_number < 1:
            raise ValueError(f"page_number is 1-indexed, got {self.page_number}")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def is_empty(self) -> bool:
        """True when the page yielded no extractable text — a blank or scanned page."""
        return not self.text.strip()
