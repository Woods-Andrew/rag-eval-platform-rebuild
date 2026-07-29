"""Persisting corpus embeddings between runs."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray

__all__ = ["DEFAULT_CACHE_DIR", "EmbeddingCache", "encoder_identity", "fingerprint"]

DEFAULT_CACHE_DIR = Path(".cache/embeddings")


def fingerprint(model: str, chunk_ids: Sequence[str]) -> str:
    """Derive a cache key from the model and the exact chunks it embedded.

    Chunk IDs already carry a digest of their own text, so hashing them covers the
    document *and* the chunking configuration: re-chunking, editing the PDF, or
    reordering the corpus all produce a different key. The model name is included
    because two encoders will happily produce same-shaped vectors that mean nothing to
    each other.
    """
    digest = hashlib.sha256(model.encode("utf-8"))
    for chunk_id in chunk_ids:
        digest.update(b"\x00")
        digest.update(chunk_id.encode("utf-8"))
    return digest.hexdigest()[:16]


def encoder_identity(encoder: object) -> str:
    """Name an encoder for cache purposes, preferring its model name.

    Falls back to the class name for encoders that do not expose one — fakes, mostly.
    A weak identity is tolerable because the stored chunk IDs are verified on load, so
    a collision produces a miss rather than wrong vectors.
    """
    model_name = getattr(encoder, "model_name", None)
    return model_name if isinstance(model_name, str) and model_name else type(encoder).__name__


@dataclass(frozen=True)
class EmbeddingCache:
    """A directory of embedding matrices, keyed by model and corpus.

    Embedding a corpus is the slowest thing this project does and the most wasteful to
    repeat: the vectors depend only on the text and the model, neither of which changes
    between runs. The in-process rule is "encode the corpus once"; this extends it
    across processes.

    Every entry stores the chunk IDs it was built from and they are checked on load, so
    a stale or colliding entry is a **miss**, never silently wrong vectors. That check
    is the reason this is safe to enable by default: the worst case is recomputation.
    """

    directory: Path = DEFAULT_CACHE_DIR

    def path_for(self, key: str) -> Path:
        return Path(self.directory) / f"{key}.npz"

    def load(self, key: str, chunk_ids: Sequence[str]) -> NDArray[np.float32] | None:
        """Return cached embeddings for ``chunk_ids``, or ``None`` on any miss.

        A corrupt or unreadable entry is a miss too. A cache is an optimization, and an
        optimization that can crash the pipeline is worse than no cache at all.
        """
        path = self.path_for(key)
        if not path.is_file():
            return None

        try:
            with np.load(path, allow_pickle=False) as stored:
                embeddings = np.asarray(stored["embeddings"], dtype=np.float32)
                cached_ids = [str(chunk_id) for chunk_id in stored["chunk_ids"]]
        except (OSError, ValueError, KeyError):
            return None

        if cached_ids != list(chunk_ids) or embeddings.shape[0] != len(cached_ids):
            return None
        return embeddings

    def store(
        self, key: str, chunk_ids: Sequence[str], embeddings: NDArray[np.float32]
    ) -> Path | None:
        """Write embeddings to the cache, returning ``None`` if the write failed.

        Written to a temporary file and moved into place, so an interrupted run leaves
        no half-written entry for the next one to read.
        """
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError(
                f"cannot cache {embeddings.shape[0]} vectors for {len(chunk_ids)} chunks"
            )

        path = self.path_for(key)
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written through a file handle because np.savez appends ".npz" to a bare
            # path, which would defeat the atomic rename below.
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    embeddings=np.asarray(embeddings, dtype=np.float32),
                    chunk_ids=np.asarray(list(chunk_ids)),
                )
            temporary.replace(path)
        except OSError:
            # The cleanup can fail for the same reason the write did — an unwritable
            # or non-existent parent — so it must not be able to raise either.
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return path
