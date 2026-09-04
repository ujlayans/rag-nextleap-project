"""Numpy-backed vector store used at runtime (no ChromaDB, no torch).

The Chroma collection is exported once into ``data/retrieval/`` — float32
vectors in ``index.npz`` and row-aligned ``records.json``. Runtime queries are
a simple dot product over the (tiny) matrix; this keeps the deployed app well
under Render's 512 MB free-tier limit that ChromaDB + torch blow through.

Query output is shaped exactly like ChromaDB's ``query()`` result so the
retriever code is unchanged ("ids", "distances", "metadatas", "documents").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DATA_DIR, RETRIEVAL_INDEX_PATH, RETRIEVAL_RECORDS_PATH

logger = logging.getLogger(__name__)


class NumpyVectorStore:
    """In-memory cosine search over exported vectors.

    Only depends on ``numpy`` — no chromadb, grpc, or sqlite — so importing and
    holding it is cheap relative to ChromaDB.

    Parameters
    ----------
    index_path:
        ``.npz`` with an aligned ``vectors`` float32 array.
    records_path:
        JSON with ``ids``, ``documents``, ``metadatas`` (same row order).
    """

    def __init__(
        self,
        index_path: Path = RETRIEVAL_INDEX_PATH,
        records_path: Path = RETRIEVAL_RECORDS_PATH,
    ) -> None:
        self.index_path = Path(index_path)
        self.records_path = Path(records_path)
        self._vectors: np.ndarray | None = None
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.index_path.is_file() or not self.records_path.is_file():
            raise FileNotFoundError(
                f"Runtime index missing: {self.index_path} / {self.records_path}"
            )
        with np.load(self.index_path) as data:
            self._vectors = np.asarray(data["vectors"], dtype=np.float32)
        records = json.loads(self.records_path.read_text(encoding="utf-8"))
        self._ids = records["ids"]
        self._documents = records["documents"]
        self._metadatas = [dict(m) for m in records["metadatas"]]
        if len(self._ids) != len(self._vectors):
            raise ValueError("records.json row count != vectors.npz rows")

    def count(self) -> int:
        return len(self._ids)

    @classmethod
    def available(cls) -> bool:
        return RETRIEVAL_INDEX_PATH.is_file() and RETRIEVAL_RECORDS_PATH.is_file()

    def exists(self) -> bool:
        return cls.available()

    def close(self) -> None:
        """No-op for API parity with VectorStore."""

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
        """Top ``k`` nearest vectors to *query_embedding* (cosine distance).

        Returns a chroma-shaped dict: ``ids``/``distances``/``metadatas`` and
        (when asked) ``documents``, each a list of lists (single query row).
        """
        q = np.atleast_2d(query_embedding).astype(np.float32)
        if self._vectors is None or len(self._vectors) == 0:
            return _empty_result(include_documents)

        scores = q @ self._vectors.T  # (1, N) dot product on normalized vectors
        order = np.argsort(-scores[0])  # highest cosine first

        chosen_ids: list[str] = []
        chosen_scores: list[float] = []
        for idx in order:
            if where and not _matches_where(self._metadatas[idx], where):
                continue
            chosen_ids.append(self._ids[idx])
            chosen_scores.append(float(1.0 - scores[0, idx]))
            if len(chosen_ids) >= k:
                break

        result: dict[str, Any] = {
            "ids": [chosen_ids],
            "distances": [chosen_scores],
            "metadatas": [
                [self._metadatas[self._ids.index(cid)] for cid in chosen_ids]
            ],
        }
        if include_documents:
            result["documents"] = [
                [self._documents[self._ids.index(cid)] for cid in chosen_ids]
            ]
        return result

    @staticmethod
    def resolve_scheme_where(scheme_name: str | None) -> dict | None:
        if scheme_name:
            return {"scheme_name": scheme_name}
        return None


def _matches_where(meta: dict[str, Any], where: dict) -> bool:
    for key, value in where.items():
        if meta.get(key) != value:
            return False
    return True


def export_from_chroma() -> int:
    """(Re)build ``data/retrieval/`` runtime artifacts from the Chroma store.

    Called by the vector-store build so the deployed app only needs the tiny
    numpy index + records (never ChromaDB at runtime). Returns record count.
    """
    from src.config import RETRIEVAL_DIR, RETRIEVAL_INDEX_PATH, RETRIEVAL_RECORDS_PATH
    from src.vector_store.store import VectorStore

    store = VectorStore()
    records = store.collection.get(include=["embeddings", "documents", "metadatas"])
    ids = records["ids"]
    vectors = np.asarray(records["embeddings"], dtype=np.float32)
    if len(ids) != len(vectors):
        raise ValueError("embeddings/ids length mismatch in Chroma collection")

    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RETRIEVAL_INDEX_PATH, vectors=vectors)
    RETRIEVAL_RECORDS_PATH.write_text(
        json.dumps(
            {
                "ids": ids,
                "documents": records["documents"],
                "metadatas": records["metadatas"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(
        "Exported runtime index (%d vectors) -> %s",
        len(ids),
        RETRIEVAL_INDEX_PATH.parent,
    )
    try:
        store.close()
    except Exception:  # pragma: no cover - defensive
        pass
    return len(ids)


def _empty_result(include_documents: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ids": [[]],
        "distances": [[]],
        "metadatas": [[]],
    }
    if include_documents:
        result["documents"] = [[]]
    return result