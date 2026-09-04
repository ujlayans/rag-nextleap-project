"""Phase 2: Chunking — split snapshots into citation-ready chunks."""

from src.chunking.schemas import Chunk
from src.chunking.splitter import chunk_markdown

__all__ = ["Chunk", "chunk_markdown"]
