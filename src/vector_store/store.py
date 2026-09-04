"""ChromaDB-backed vector store: persist vectors + metadata, serve similarity search.

Design note: embeddings are supplied *explicitly* (not computed by ChromaDB) to
keep every vector and query produced by the same frozen MiniLM model (PRD §7,
Phase 3 guardrail). We encode queries with the same model used to embed chunks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from src.config import COLLECTION_NAME, CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)

# ChromaDB's SharedSystemClient is not thread-safe (can raise
# `AttributeError: bindings` when the same path's client is created/released
# across threads, e.g. Streamlit reruns). Disable telemetry to reduce the
# shared-system bookkeeping involved in that failure mode.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# Chroma metadata values must be str/int/float/bool — every field used below.
_METADATA_FIELDS = (
    "scheme_name",
    "scheme_category",
    "source_url",
    "snapshot_date",
    "section",
)


class VectorStore:
    """CRUD + query wrapper around a single persistent Chroma collection.

    Parameters
    ----------
    persist_dir:
        Directory for ChromaDB on-disk persistence. Defaults to the
        architecture-defined ``data/vectordb/chroma``.
    collection_name:
        Collection to read/write. Defaults to ``sbi_schemes``.
    """

    def __init__(
        self,
        persist_dir: Path = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = None  # lazily resolved

    # ------------------------------------------------------------------
    # Collection access
    # ------------------------------------------------------------------

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def count(self) -> int:
        return self.collection.count()

    def exists(self) -> bool:
        names = {c.name for c in self._client.list_collections()}
        return self.collection_name in names

    def close(self) -> None:
        """Release the underlying Chroma client.

        Explicitly closing avoids Chroma's shared-system teardown racing on GC,
        which can surface as ``AttributeError: bindings`` in threaded contexts
        such as the Streamlit ScriptRunner.
        """
        self._collection = None
        try:
            self._client.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Chroma close warning: %s", exc)

    # ------------------------------------------------------------------
    # Ingest / reset
    # ------------------------------------------------------------------

    def rebuild(
        self,
        embedding_function=None,
        metadata: dict | None = None,
    ) -> None:
        """Drop and recreate the collection from scratch."""
        try:
            self._client.delete_collection(self.collection_name)
            logger.info("Dropped existing collection %r", self.collection_name)
        except Exception:
            logger.debug("No existing collection %r to drop", self.collection_name)

        self.collection  # triggers (re)creation
        if metadata:
            self._client.modify_collection(
                name=self.collection_name, metadata=metadata
            )

    def upsert(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Upsert vectors + documents + metadata. Idempotent on ``ids``.

        Returns the number of records added.
        """
        if not ids:
            return 0

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=[_clean_metadata(m) for m in metadatas],
        )
        return len(ids)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 4,
        where: dict | None = None,
        include_documents: bool = True,
    ) -> dict[str, Any]:
        """Query the collection for the ``k`` nearest vectors to *query_embedding*.

        Parameters
        ----------
        query_embedding:
            ``(dim,)`` or ``(1, dim)`` L2-normalized query vector.
        k:
            Number of nearest neighbours to return.
        where:
            Optional Chroma metadata filter (e.g. ``{"scheme_name": "..."}``).
        include_documents:
            Whether to return the source document text.

        Returns
        -------
        ChromaDB ``query`` result dict with keys ``ids``, ``distances``,
        ``metadatas``, and (optionally) ``documents``.
        """
        q = np.atleast_2d(query_embedding).astype(np.float32)
        include = ["metadatas", "distances"]
        if include_documents:
            include.append("documents")

        return self.collection.query(
            query_embeddings=q.tolist(),
            n_results=k,
            where=where,
            include=include,
        )

    @staticmethod
    def resolve_scheme_where(scheme_name: str | None) -> dict | None:
        """Build a Chroma ``where`` filter dict for a scheme name, or ``None``."""
        if scheme_name:
            return {"scheme_name": scheme_name}
        return None


def _clean_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Restrict metadata to plain, Chroma-serializable scalar fields."""
    cleaned: dict[str, Any] = {}
    for key in _METADATA_FIELDS:
        if key in meta and meta[key] is not None:
            cleaned[key] = meta[key]
    return cleaned
