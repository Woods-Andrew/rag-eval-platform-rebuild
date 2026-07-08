"""RAG evaluation platform for technical documents.

The package is organized around a retrieval pipeline that keeps document
provenance (source, page, chunk ID) intact from PDF ingestion through to
cited answer generation, and an evaluation pipeline that measures retrieval
quality independently of generation.

Subpackages are added as milestones land; see README.md for current status.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
