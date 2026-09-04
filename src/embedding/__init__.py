"""Phase 3: Embedding — convert chunks into dense vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.embedding.encoder import Embedder
    from src.embedding.onnx_embedder import OnnxMiniLMEmbedder

__all__ = ["Embedder", "OnnxMiniLMEmbedder"]


def __getattr__(name: str):
    if name == "Embedder":
        from src.embedding.encoder import Embedder

        return Embedder
    if name == "OnnxMiniLMEmbedder":
        from src.embedding.onnx_embedder import OnnxMiniLMEmbedder

        return OnnxMiniLMEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")