"""Vector store package: lightweight runtime store + ChromaDB build store.

ChromaDB ``VectorStore`` is intentionally NOT imported at package import time:
it is only needed to build/export the index, and importing it pulls in tens of
MBs of resident memory (Render free tier = 512 MB). Access it explicitly via
``from src.vector_store.store import VectorStore``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.vector_store.numpy_store import NumpyVectorStore
    from src.vector_store.store import VectorStore

__all__ = ["NumpyVectorStore", "VectorStore"]


def __getattr__(name: str):
    if name == "VectorStore":
        from src.vector_store.store import VectorStore

        return VectorStore
    if name == "NumpyVectorStore":
        from src.vector_store.numpy_store import NumpyVectorStore

        return NumpyVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")