"""CLI entry point for Phase 2: Chunking.

Usage
-----
    python -m src.chunking.run                 # chunk all schemes in data/raw/
    python -m src.chunking.run --slug NAME     # chunk a single scheme
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking.splitter import chunk_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
CHUNKS_DIR = ROOT / "data" / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "chunks.jsonl"
MANIFEST_FILE = CHUNKS_DIR / "manifest.json"


def _discover_schemes() -> list[tuple[str, Path]]:
    """Return [(slug, md_path)] in sorted order from data/raw/*.md."""
    schemes: list[tuple[str, Path]] = []
    for md in sorted(RAW_DIR.glob("*.md")):
        schemes.append((md.stem, md))
    return schemes


def chunk_schemes(slug_filter: str | None = None) -> list[dict]:
    """Chunk all discovered schemes (or a single slug). Returns list of chunk dicts."""
    schemes = _discover_schemes()
    all_chunks: list[dict] = []

    for slug, md_path in schemes:
        if slug_filter and slug != slug_filter:
            continue

        md_text = md_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(md_text, slug)
        all_chunks.extend([c.to_dict() for c in chunks])
        logger.info(
            "Chunked %s: %d chunk(s) → sections %s",
            slug,
            len(chunks),
            sorted({c.section for c in chunks}),
        )

    return all_chunks


def write_outputs(chunks: list[dict]) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Build + write manifest
    by_scheme: dict[str, dict] = {}
    for c in chunks:
        slug = c["chunk_id"].split("__")[0]
        entry = by_scheme.setdefault(
            slug, {"chunk_count": 0, "sections": set()}
        )
        entry["chunk_count"] += 1
        entry["sections"].add(c["section"])
    manifest = {
        "snapshot_date": chunks[0]["snapshot_date"] if chunks else None,
        "total_chunks": len(chunks),
        "schemes": {
            slug: {
                "chunk_count": e["chunk_count"],
                "sections": sorted(e["sections"]),
            }
            for slug, e in sorted(by_scheme.items())
        },
    }
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(slug_filter: str | None = None) -> int:
    logger.info("Phase 2 — Chunking")
    chunks = chunk_schemes(slug_filter)
    write_outputs(chunks)
    logger.info("Wrote %d chunk(s) → %s", len(chunks), CHUNKS_FILE)
    logger.info("Manifest → %s", MANIFEST_FILE)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Split markdown snapshots into citation-ready chunks."
    )
    parser.add_argument(
        "--slug",
        help="Chunk only this scheme (e.g. sbi-small-cap-fund). Default: all.",
    )
    args = parser.parse_args()

    count = run(slug_filter=args.slug)
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
