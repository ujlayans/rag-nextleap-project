"""Retriever: embed a question and fetch the top-k chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import TOP_K
from src.embedding.onnx_embedder import get_embedder
from src.vector_store.numpy_store import NumpyVectorStore

if TYPE_CHECKING:  # heavy deps are imported lazily (runtime stays < 512MB)
    from src.embedding.encoder import Embedder
    from src.vector_store.store import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    scheme_name: str
    scheme_category: str
    source_url: str
    snapshot_date: str
    section: str


class Retriever:
    """Embeds a query (same MiniLM model) and returns ranked chunks."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        top_k: int = TOP_K,
    ) -> None:
        self.embedder = embedder or get_embedder()
        self.store = store or _build_store()
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        *,
        scheme_name: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Embed *question* and query the store, optionally filtered by scheme.

        Scores are cosine distance (lower = more similar) because vectors are
        L2-normalized and the collection uses cosine space.
        """
        k = top_k or self.top_k
        qvec: np.ndarray = self.embedder.encode_query(question)
        where = self.store.resolve_scheme_where(scheme_name)
        result = self.store.query(qvec, k=k, where=where)

        ids = result["ids"][0]
        distances = result["distances"][0]
        metadatas = result["metadatas"][0]
        documents = result.get("documents", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for cid, dist, meta, doc in zip(ids, distances, metadatas, documents):
            meta = meta or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=doc or "",
                    score=float(dist),
                    scheme_name=meta.get("scheme_name", ""),
                    scheme_category=meta.get("scheme_category", ""),
                    source_url=meta.get("source_url", ""),
                    snapshot_date=meta.get("snapshot_date", ""),
                    section=meta.get("section", ""),
                )
            )
        return chunks


def _build_store() -> Any:
    """Prefer the lightweight numpy store; fall back to ChromaDB if needed."""
    if NumpyVectorStore.available():
        return NumpyVectorStore()
    from src.vector_store.store import VectorStore  # lazy: pulls chromadb

    return VectorStore()
