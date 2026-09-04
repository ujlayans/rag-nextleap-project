"""HTML → clean Markdown parser with YAML front-matter.

In addition to trafilatura's article-text extraction, this parser taps the
Groww page's embedded ``__NEXT_DATA__`` JSON to recover structured facts
(lock-in, SIP minimum, exit load, expense ratio, risk, NAV, AUM, benchmark…)
that are otherwise dropped by text-only extraction.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date

import trafilatura
from trafilatura.settings import use_config as trafilatura_config

logger = logging.getLogger(__name__)

# Category mapping: slug → human-readable category (PRD §4)
_CATEGORY_MAP: dict[str, str] = {
    "sbi-small-cap-fund": "Small Cap",
    "sbi-mid-cap-fund": "Mid Cap",
    "sbi-large-cap-fund": "Large Cap",
    "sbi-nifty-next-50": "Nifty Index",
    "sbi-gold-fund": "Gold",
    "sbi-elss-tax-saver": "ELSS",
    "sbi-flexicap-fund": "Flexi Cap",
}

# Scheme name mapping: slug → full name as on Groww
_SCHEME_NAME_MAP: dict[str, str] = {
    "sbi-small-cap-fund": "SBI Small Cap Fund Direct Growth",
    "sbi-mid-cap-fund": "SBI Mid Cap Direct Plan Growth",
    "sbi-large-cap-fund": "SBI Large Cap Direct Plan Growth",
    "sbi-nifty-next-50": "SBI Nifty Next 50 Index Fund Direct Growth",
    "sbi-gold-fund": "SBI Gold Direct Plan Growth",
    "sbi-elss-tax-saver": "SBI ELSS Tax Saver Fund Direct Growth",
    "sbi-flexicap-fund": "SBI Flexicap Fund Direct Growth",
}

# Trafilatura config: focus on main content, strip boilerplate
_config = trafilatura_config()
_config.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")
_config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "200")


def parse_html_to_markdown(
    html: str,
    *,
    slug: str,
    source_url: str,
    snapshot_date: str | None = None,
) -> str:
    """Extract main content from *html* and return Markdown with YAML front-matter.

    Parameters
    ----------
    html:
        Raw HTML string from Groww.
    slug:
        Scheme slug (e.g. ``"sbi-small-cap-fund"``). Used for metadata lookups.
    source_url:
        Original Groww URL for this scheme.
    snapshot_date:
        ISO date string. Defaults to today.

    Returns
    -------
    str
        Markdown document with YAML front-matter block.
    """
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    scheme_name = _SCHEME_NAME_MAP.get(slug, slug)
    category = _CATEGORY_MAP.get(slug, "Unknown")

    # Extract main content using trafilatura
    extracted = trafilatura.extract(
        html,
        config=_config,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=False,
        favor_recall=True,
        output_format="txt",
    )

    if not extracted:
        logger.warning("trafilatura returned empty for %s — falling back to raw text", slug)
        extracted = _fallback_extract(html)

    # Recover structured facts from the embedded __NEXT_DATA__ payload (fields
    # that trafilatura's text extraction would otherwise discard, e.g. lock-in).
    facts_block = _extract_key_facts(html)

    # Build YAML front-matter
    frontmatter = (
        "---\n"
        f"scheme_name: \"{scheme_name}\"\n"
        f"category: \"{category}\"\n"
        f"source_url: \"{source_url}\"\n"
        f"snapshot_date: \"{snapshot_date}\"\n"
        "---\n\n"
    )

    if facts_block:
        return frontmatter + extracted + "\n\n" + facts_block
    return frontmatter + extracted


# ---------------------------------------------------------------------------
# Structured facts from Groww's __NEXT_DATA__ (server-rendered JSON state)
# ---------------------------------------------------------------------------

# field path -> human-readable label; nested (a.b.c) supported via traversal.
_FACT_PATHS: dict[str, str] = {
    "lock_in": "Lock-in",
    "exit_load": "Exit load",
    "min_sip_investment": "Minimum SIP investment",
    "min_investment_amount": "Minimum investment amount",
    "expense_ratio": "Expense ratio",
    "risk": "Risk",
    "return_stats.0.risk": "Risk",
    "risk_rating": "Risk rating",
    "return_stats.0.risk_rating": "Risk rating",
    "groww_rating": "Groww rating",
    "aum": "AUM (in Cr)",
    "benchmark_name": "Benchmark",
    "nav": "Latest NAV",
    "nav_date": "NAV date",
    "launch_date": "Launch date",
    "fund_manager_details": "Fund manager",
    "sub_category": "Sub-category",
}

# lock_in is an object {years, months, days} — render as "3 years" style text.
def _fmt_lock_in(value) -> str | None:
    if not isinstance(value, dict):
        return None
    years = value.get("years", 0)
    months = value.get("months", 0)
    days = value.get("days", 0)
    if not (years or months or days):
        return None
    parts = []
    if years:
        parts.append(f"{years} year{'s' if years != 1 else ''}")
    if months:
        parts.append(f"{months} month{'s' if months != 1 else ''}")
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    return ", ".join(parts)


def _parse_next_data(html: str) -> dict | None:
    """Return the mfServerSideData dict from the __NEXT_DATA__ script, or None."""
    m = re.search(
        r'<script id="__NEXT_DATA__"[^>]*type="application/json"[^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    try:
        return payload["props"]["pageProps"]["mfServerSideData"]
    except (KeyError, TypeError):
        return None


def _lookup(data: dict, dotted: str):
    """Traverse a dotted path through nested dicts/lists; return value or None."""
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
        if cur is None:
            return None
    return cur


def _fund_manager_names(data: dict) -> str | None:
    details = data.get("fund_manager_details") or data.get("fund_manager")
    if isinstance(details, str):
        return details or None
    if isinstance(details, list):
        names = [
            d.get("person_name")
            for d in details
            if isinstance(d, dict) and d.get("person_name")
        ]
        return ", ".join(names) or None
    return None


def _extract_key_facts(html: str) -> str:
    """Return a Markdown 'Key facts / snapshot' block, or '' if not available."""
    data = _parse_next_data(html)
    if not data:
        return ""

    rows: list[str] = []
    seen_labels: set[str] = set()
    for dotted, label in _FACT_PATHS.items():
        if dotted == "fund_manager_details":
            value = _fund_manager_names(data)
        else:
            value = _lookup(data, dotted)
        if value is None or value == "" or value == []:
            continue
        if label in seen_labels:
            continue  # avoid duplicate rows when multiple paths map to one label
        seen_labels.add(label)
        if dotted == "lock_in":
            rendered = _fmt_lock_in(value)
            if not rendered:
                continue
        elif isinstance(value, (dict, list)):
            continue  # skip any other complex objects we don't model
        else:
            rendered = str(value).strip()

        rows.append(f"- {label}: {rendered}")

    if not rows:
        return ""

    return "# Key facts (from Groww page snapshot)\n\n" + "\n".join(rows)


def _fallback_extract(html: str) -> str:
    """Minimal fallback: strip tags with BeautifulSoup if trafilatura fails."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, nav, footer, header
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        import re

        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    except ImportError:
        logger.error("BeautifulSoup not installed — cannot fallback-parse HTML")
        return ""


def parse_snapshot_file(
    html_path: str | None = None,
    *,
    slug: str,
    source_url: str,
    snapshot_date: str | None = None,
) -> str:
    """Convenience wrapper: read an HTML file from disk and parse it."""
    if html_path is None:
        from pathlib import Path

        from src.config import ROOT

        html_path = str(ROOT / "data" / "raw" / f"{slug}.html")

    html = Path(html_path).read_text(encoding="utf-8")
    return parse_html_to_markdown(
        html, slug=slug, source_url=source_url, snapshot_date=snapshot_date
    )
