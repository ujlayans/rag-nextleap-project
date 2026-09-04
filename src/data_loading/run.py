"""CLI entry point for Phase 1: Data Loading.

Usage
-----
    python -m src.data_loading.run              # fetch + parse (skip if today's snapshots exist)
    python -m src.data_loading.run --force      # re-fetch everything
    python -m src.data_loading.run --dry-run    # fetch only, skip parse
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loading.fetcher import SCHEME_URLS, fetch_scheme_pages
from src.data_loading.parser import parse_html_to_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"


def run(force: bool = False, dry_run: bool = False) -> int:
    """Run the full Phase 1 pipeline. Returns number of schemes processed."""
    logger.info("Phase 1 — Data Loading")
    logger.info("Output directory: %s", RAW_DIR)

    # Step 1: Fetch raw HTML
    results = fetch_scheme_pages(RAW_DIR, force=force)
    logger.info("Fetched %d scheme page(s)", len(results))

    if dry_run:
        logger.info("--dry-run: skipping HTML → Markdown parse")
        return len(results)

    # Step 2: Parse each HTML file to Markdown with front-matter
    parsed_count = 0
    for result in results:
        try:
            markdown = parse_html_to_markdown(
                result.html,
                slug=result.slug,
                source_url=result.url,
            )

            # Save as .md alongside the .html
            md_path = RAW_DIR / f"{result.slug}.md"
            md_path.write_text(markdown, encoding="utf-8")
            logger.info("Parsed %s → %s", result.slug, md_path)
            parsed_count += 1

        except Exception as exc:
            logger.error("Failed to parse %s: %s", result.slug, exc)

    logger.info("Phase 1 complete: %d fetched, %d parsed", len(results), parsed_count)
    return parsed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: Fetch and parse Groww scheme pages."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all pages even if today's snapshots exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch raw HTML only; skip Markdown parsing.",
    )
    args = parser.parse_args()

    count = run(force=args.force, dry_run=args.dry_run)
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
