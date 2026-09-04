"""CLI entry point for Phase 3: Embedding.

Reads chunks from ``data/chunks/chunks.jsonl``, encodes each chunk's text with
the MiniLM model, and persists the vectors to ``data/embeddings/``.

Outputs
-------
- data/embeddings/vectors.npy     (N × 384) float32 L2-normalized array
- data/embeddings/chunk_ids.json  ordered list mapping row index → chunk_id
- data/embeddings/embedding_meta.json  model + dim + timestamp info

Usage
-----
    python -m src.embedding.run
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

from src.config import EMBEDDING_MODEL
from src.embedding.encoder import Embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
EMBED_DIR = ROOT / "data" / "embeddings"
VECTORS_FILE = EMBED_DIR / "vectors.npy"
CHUNK_IDS_FILE = EMBED_DIR / "chunk_ids.json"
META_FILE = EMBED_DIR / "embedding_meta.json"


def load_chunks(chunks_file: Path = CHUNKS_FILE) -> list[dict]:
    """Read chunks.jsonl into a list of chunk dicts (preserving order)."""
    chunks: list[dict] = []
    with chunks_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_chunks(
    chunks: list[dict],
    *,
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    device: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Embed chunk texts. Returns ``(vectors, chunk_ids)`` aligned by index."""
    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    embedder = Embedder(model_name=model_name, device=device, batch_size=batch_size)
    logger.info("Encoding %d chunk(s) in batches of %d …", len(texts), batch_size)
    vectors = embedder.encode(texts, batch_size=batch_size)
    return vectors, chunk_ids


def write_outputs(vectors: np.ndarray, chunk_ids: list[str], model_name: str) -> None:
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(VECTORS_FILE, vectors.astype(np.float32))

    CHUNK_IDS_FILE.write_text(
        json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "model_name": model_name,
        "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "num_vectors": int(vectors.shape[0]),
        "normalized_l2": True,
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def run(
    *,
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    device: str | None = None,
) -> int:
    logger.info("Phase 3 — Embedding")
    chunks = load_chunks()
    if not chunks:
        logger.error("No chunks found at %s. Run Phase 2 first.", CHUNKS_FILE)
        return 0

    vectors, chunk_ids = embed_chunks(
        chunks, model_name=model_name, batch_size=batch_size, device=device
    )
    write_outputs(vectors, chunk_ids, model_name)

    logger.info(
        "Wrote %d × %d vectors → %s",
        vectors.shape[0],
        vectors.shape[1],
        VECTORS_FILE,
    )
    logger.info("Chunk id map → %s", CHUNK_IDS_FILE)
    return vectors.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Embed chunks into dense vectors."
    )
    parser.add_argument("--model", default=EMBEDDING_MODEL, help="HF model id.")
    parser.add_argument("--batch-size", type=int, default=64, help="Encode batch size.")
    parser.add_argument("--device", default=None, help="torch device (cpu/mps/cuda).")
    args = parser.parse_args()

    count = run(model_name=args.model, batch_size=args.batch_size, device=args.device)
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
