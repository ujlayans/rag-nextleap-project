"""CLI entry point for Phase 4: Vector Store (ChromaDB).

Ingests Phase 3 embeddings (``data/embeddings/vectors.npy`` + ``chunk_ids.json``)
and Phase 2 chunks (``data/chunks/chunks.jsonl``) into a persistent ChromaDB
collection, and supports querying it.

Usage
-----
    python -m src.vector_store.run                # ingest (idempotent upsert)
    python -m src.vector_store.run --rebuild      # drop + recreate collection
    python -m src.vector_store.run --count        # print number of vectors
    python -m src.vector_store.run --query "What is the expense ratio of SBI Small Cap?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    CHUNK_IDS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTORS_PATH,
)
from src.embedding.encoder import Embedder
from src.vector_store.store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def ingest(rebuild: bool = False) -> int:
    """Load Phase 2/3 artifacts into ChromaDB. Returns number of vectors."""
    chunks = load_chunks()
    if not chunks:
        logger.error("No chunks at %s. Run Phases 2 & 3 first.", CHUNKS_FILE)
        return 0

    vectors = np.load(VECTORS_PATH)
    chunk_ids = json.loads(CHUNK_IDS_PATH.read_text(encoding="utf-8"))

    if vectors.shape[0] != len(chunks) or len(chunk_ids) != len(chunks):
        logger.error(
            "Mismatch: vectors=%d, ids=%d, chunks=%d. Re-run Phases 2/3.",
            vectors.shape[0],
            len(chunk_ids),
            len(chunks),
        )
        return 0

    store = VectorStore()

    if rebuild:
        store.rebuild()

    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "scheme_name": c["scheme_name"],
            "scheme_category": c["scheme_category"],
            "source_url": c["source_url"],
            "snapshot_date": c["snapshot_date"],
            "section": c["section"],
        }
        for c in chunks
    ]

    n = store.upsert(vectors, chunk_ids, documents, metadatas)
    logger.info(
        "Upserted %d vector(s) into %s (%s). Total: %d",
        n,
        CHROMA_PERSIST_DIR_NAME(),
        COLLECTION_NAME,
        store.count(),
    )
    return n


def CHROMA_PERSIST_DIR_NAME() -> str:
    from src.config import CHROMA_PERSIST_DIR

    return str(CHROMA_PERSIST_DIR)


def run_query(query_text: str, k: int = 4, scheme_name: str | None = None) -> dict:
    embedder = Embedder(model_name=EMBEDDING_MODEL)
    store = VectorStore()
    where = VectorStore.resolve_scheme_where(scheme_name)
    qvec = embedder.encode_query(query_text)
    return store.query(qvec, k=k, where=where)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Build and query the ChromaDB vector store."
    )
    parser.add_argument("--rebuild", action="store_true", help="Drop + recreate collection.")
    parser.add_argument("--count", action="store_true", help="Print vector count and exit.")
    parser.add_argument("--query", help="Query the store with this text and print top-k.")
    parser.add_argument("--k", type=int, default=4, help="Top-k for --query.")
    parser.add_argument("--scheme", help="Restrict --query to one scheme name.")
    args = parser.parse_args()

    if args.count:
        store = VectorStore()
        print(f"{store.count()}")
        sys.exit(0)

    if args.query:
        result = run_query(args.query, k=args.k, scheme_name=args.scheme)
        ids = result["ids"][0]
        distances = result["distances"][0]
        metas = result["metadatas"][0]
        docs = result.get("documents", [[]])[0]
        for i, (cid, dist, meta, doc) in enumerate(zip(ids, distances, metas, docs), 1):
            print(f"\n#{i}  {cid}  (dist={dist:.4f})  [{meta.get('section')}]")
            print(f"  scheme: {meta.get('scheme_name')}")
            print(f"  url:    {meta.get('source_url')}")
            snippet = (doc or "").replace("\n", " ")[:160]
            print(f"  doc:    {snippet}…")
        sys.exit(0)

    n = ingest(rebuild=args.rebuild)
    sys.exit(0 if n > 0 else 1)


if __name__ == "__main__":
    main()
