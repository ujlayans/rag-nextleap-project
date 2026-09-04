"""HTTP fetcher for Groww scheme pages with retry and URL validation."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import NamedTuple

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PRD §4 — closed corpus: only these 7 URLs are allowed
# ---------------------------------------------------------------------------

SCHEME_URLS: dict[str, str] = {
    "sbi-small-cap-fund": "https://groww.in/mutual-funds/sbi-small-midcap-fund-direct-growth",
    "sbi-mid-cap-fund": "https://groww.in/mutual-funds/sbi-mid-cap-direct-plan-growth",
    "sbi-large-cap-fund": "https://groww.in/mutual-funds/sbi-large-cap-direct-plan-growth",
    "sbi-nifty-next-50": "https://groww.in/mutual-funds/sbi-nifty-next-50-index-fund-direct-growth",
    "sbi-gold-fund": "https://groww.in/mutual-funds/sbi-gold-fund-direct-growth",
    "sbi-elss-tax-saver": "https://groww.in/mutual-funds/sbi-elss-tax-saver-fund-direct-growth",
    "sbi-flexicap-fund": "https://groww.in/mutual-funds/sbi-flexicap-fund-direct-growth",
}

ALLOWED_HOSTS = {"groww.in", "www.groww.in"}


class FetchResult(NamedTuple):
    slug: str
    url: str
    html: str
    status_code: int


def _validate_url(url: str) -> bool:
    """Reject any URL not in the approved corpus."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_HOSTS:
        logger.warning("Rejected URL (host not allowed): %s", url)
        return False
    if not url.startswith("https://"):
        logger.warning("Rejected URL (not HTTPS): %s", url)
        return False
    return True


def _validate_slug(slug: str) -> bool:
    """Reject any slug not in the approved corpus."""
    if slug not in SCHEME_URLS:
        logger.warning("Rejected slug (not in corpus): %s", slug)
        return False
    return True


def fetch_scheme_pages(
    output_dir: Path,
    *,
    force: bool = False,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> list[FetchResult]:
    """Fetch all 7 Groww scheme pages and save raw HTML to *output_dir*.

    Parameters
    ----------
    output_dir:
        Directory to write ``{slug}.html`` files into (created if needed).
    force:
        If ``False`` and a file already exists for today, skip that fetch.
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        Number of retry attempts per request (exponential back-off).

    Returns
    -------
    list[FetchResult]
        One result per scheme that was fetched (skipped schemes are omitted).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    results: list[FetchResult] = []

    for slug, url in SCHEME_URLS.items():
        if not _validate_slug(slug):
            continue
        if not _validate_url(url):
            continue

        dest = output_dir / f"{slug}.html"

        # Dedup: skip if file exists and was written today (unless --force)
        if not force and dest.exists():
            mtime = date.fromtimestamp(dest.stat().st_mtime).isoformat()
            if mtime == today:
                logger.info("Skipping %s (already fetched today)", slug)
                continue

        html = _fetch_with_retry(url, timeout=timeout, max_retries=max_retries)
        if html is None:
            logger.error("Failed to fetch %s after %d retries", slug, max_retries)
            continue

        dest.write_text(html, encoding="utf-8")
        logger.info("Saved %s → %s", slug, dest)
        results.append(FetchResult(slug=slug, url=url, html=html, status_code=200))

    return results


def _fetch_with_retry(
    url: str,
    *,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> str | None:
    """Fetch *url* with exponential-backoff retries. Returns HTML or ``None``."""
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                },
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.text

        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                url,
                exc,
            )
            if attempt < max_retries:
                import time

                time.sleep(2 ** attempt)  # 2s, 4s, …

    logger.error("All %d attempts failed for %s: %s", max_retries, url, last_exc)
    return None
