"""Hybrid section-aware + paragraph-based chunk splitter for Groww scheme pages."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from src.chunking.schemas import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

MAX_TOKENS = 512
MIN_TOKENS = 50
OVERLAP_TOKENS = 50

# ---------------------------------------------------------------------------
# Boilerplate to strip (nav / promo text that is not a scheme fact)
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    re.compile(r"^invest in stocks, etfs.*$", re.I),
    re.compile(r"^invest in direct mutual funds.*$", re.I),
    re.compile(r"^invest in mutual funds$", re.I),
    re.compile(r"^stocks$", re.I),
    re.compile(r"^f&o$", re.I),
    re.compile(r"^mutual funds$", re.I),
    re.compile(r"^sip$", re.I),
    re.compile(r"^view details$", re.I),
    re.compile(r"^check past data$", re.I),
    re.compile(r"^see all$", re.I),
    re.compile(r"^compare$", re.I),
    re.compile(r"^;\s*$", re.I),          # stray separators
    re.compile(r"^monthly investment$", re.I),
    re.compile(r"^rating$", re.I),
    re.compile(r"^annualised returns$", re.I),
    re.compile(r"^absolute returns$", re.I),
]

# ---------------------------------------------------------------------------
# Logical-block anchors → canonical section labels
# ---------------------------------------------------------------------------

_ANCHOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # key facts (expense ratio, NAV, AUM, min SIP) — labelled with an anchor
    ("fund_details",
     re.compile(
         r"expense ratio|fund size \(aum\)|nav:|min\.?\s*for\s*(1st|2nd|sip|investment)"
         r"|exit load|lock-in|reward|fund size",
         re.I,
     )),
    ("costs_tax",
     re.compile(
         r"exit load of \d|redeem.*(tax|taxed)|tax implication|stamp duty|"
         r"returns exceeding rs|taxed at",
         re.I,
     )),
    ("managers", re.compile(r"fund manager|current fund manager|dec \d{4} - present", re.I)),
    ("riskometer", re.compile(r"riskometer|rated (very high|high|moderate|low) risk", re.I)),
    ("objective", re.compile(r"scheme seeks|scheme aims|investment objective|aims to provide", re.I)),
    ("about", re.compile(r"is a .* mutual fund scheme launched by|made available to investors", re.I)),
    ("holdings", re.compile(r"^\| name \| sector \| instruments \| assets", re.I)),
    ("performance",
     re.compile(r"^\| name \| 3y \| 5y \| 10y|^\| over the past|fund returns|category average", re.I)),
    ("overview", re.compile(r"fund benchmark|rank \(total assets\)|date of incorporation", re.I)),
]

# ---------------------------------------------------------------------------
# Token estimation (cheap heuristic: ~4 chars per token)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). No heavy tokenizer needed."""
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


# ---------------------------------------------------------------------------
# Line-level utilities
# ---------------------------------------------------------------------------


def _is_boilerplate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for pat in _BOILERPLATE_PATTERNS:
        if pat.fullmatch(stripped):
            return True
    return False


def _is_table_line(line: str) -> bool:
    """A line that is a markdown table row (contains pipes / delimiter)."""
    return "|" in line and ("---" in line or line.strip().startswith("|"))


def _is_heading_like(line: str) -> bool:
    """Short standalone label line that starts a logical block."""
    text = line.strip()
    if not text or len(text) > 80 or "|" in text:
        return False
    # Anchors that introduce a block
    return _anchor_of(text) is not None


def _anchor_of(text: str) -> str | None:
    """Return the canonical section label if *text* matches an anchor, else None."""
    for label, pattern in _ANCHOR_PATTERNS:
        if pattern.search(text):
            return label
    return None


# ---------------------------------------------------------------------------
# Section object
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    label: str
    body: str


def _classify_text(text: str) -> str:
    """Best-effort canonical label for a block of text."""
    label = _anchor_of(text)
    return label or "other"


# ---------------------------------------------------------------------------
# Main chunking
# ---------------------------------------------------------------------------

_NAV_WRAPPER_PATTERNS = [
    re.compile(r"^nav:\s*\d{1,2}.*$", re.I),
    re.compile(r"^\+\d+(\.\d+)?%\s*$"),       # e.g. "+12.81%"
    re.compile(r"^-\d+(\.\d+)?%\s*$"),        # e.g. "-0.30%"
    re.compile(r"^\d+[yd]\s*(annualised)?\s*$", re.I),  # "3Y annualised", "1D"
]


def _normalize_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned: list[str] = []
    for ln in lines:
        is_nav = any(p.fullmatch(ln) for p in _NAV_WRAPPER_PATTERNS)
        if _is_boilerplate(ln) or is_nav:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned).strip()


def _split_into_sections(body: str) -> list[_Section]:
    """Segment snapshot body into logical sections.

    Strategy:
      1. Walk lines, tracking the current ``label`` based on anchor detection.
      2. When a new anchor line appears, flush the current buffer into a section.
      3. Group consecutive lines of the same tentative label into one block,
         but allow multi-part blocks (e.g. a table following a heading).
    """
    cleaned = _normalize_text(body)
    if not cleaned:
        return []

    lines = cleaned.splitlines()
    sections: list[_Section] = []
    current_label: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_label, current_lines
        text = "\n".join(current_lines).strip()
        if text:
            label = current_label or _classify_text(text)
            sections.append(_Section(label=label, body=text))
        current_label = None
        current_lines = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Table header rows start a distinct logical block.
        table_label = _table_block_label(line)
        if table_label and current_label != table_label:
            flush()
            current_label = table_label
            current_lines.append(line)
            continue

        # Table lines: keep going under the current label.
        if _is_table_line(line):
            current_lines.append(line)
            continue

        # A heading-like anchor line starts a new block.
        label = _anchor_of(line)

        if label and (current_label is None or label != current_label):
            flush()
            current_label = label
            current_lines.append(line)
            continue

        # Regular line: append to current block.
        current_lines.append(line)
        if current_label is None:
            current_label = _classify_text("\n".join(current_lines))

    flush()
    return sections


def _table_block_label(line: str) -> str | None:
    """Return the section label if *line* is a recognized table header row."""
    lowered = line.lower()
    if lowered.startswith("| name | sector | instruments | assets"):
        return "holdings"
    if lowered.startswith("| name | 3y | 5y | 10y") or "| fund returns |" in lowered:
        return "performance"
    return None


# ---------------------------------------------------------------------------
# Long-section splitting (secondary split on paragraph / sentence boundaries)
# ---------------------------------------------------------------------------


def _split_long_section(body: str) -> list[str]:
    """Split a section exceeding MAX_TOKENS into sub-chunks, breaking on line
    boundaries, with OVERLAP_TOKENS of tail-overlap."""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0
    last_flushed_text = ""

    def flush_buffer() -> None:
        nonlocal buffer_tokens, last_flushed_text
        text = "\n".join(buffer).strip()
        if estimate_tokens(text) >= MIN_TOKENS:
            chunks.append(text)
            last_flushed_text = text
        buffer.clear()
        buffer_tokens = 0

    for i, ln in enumerate(lines):
        ln_tokens = estimate_tokens(ln)
        if buffer and (buffer_tokens + ln_tokens) > MAX_TOKENS:
            flush_buffer()
            buffer_tokens = 0
            # Overlap: carry the tail words of the last flushed line
            if i > 0:
                prev_words = lines[i - 1].split()
                overlap_text = " ".join(prev_words[-OVERLAP_TOKENS:])
                if overlap_text:
                    buffer.append(overlap_text)
                    buffer_tokens = estimate_tokens(overlap_text)
        buffer.append(ln)
        buffer_tokens += ln_tokens

    flush_buffer()
    return chunks


# ---------------------------------------------------------------------------
# Front-matter parsing
# ---------------------------------------------------------------------------


def parse_frontmatter(md_text: str) -> tuple[dict, str]:
    frontmatter: dict = {}
    body = md_text
    if md_text.startswith("---"):
        end = md_text.find("\n---", 3)
        if end != -1:
            header = md_text[3:end]
            body = md_text[end + 4 :]
            for line in header.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    frontmatter[key.strip()] = val.strip().strip('"').strip("'")
    return frontmatter, body


def chunk_markdown(md_text: str, slug: str) -> list[Chunk]:
    """Chunk a single markdown snapshot into a list of ``Chunk`` objects."""
    frontmatter, body = parse_frontmatter(md_text)

    scheme_name = frontmatter.get("scheme_name", slug)
    scheme_category = frontmatter.get("category", "Unknown")
    source_url = frontmatter.get("source_url", "")
    snapshot_date = frontmatter.get("snapshot_date", "")

    sections = _split_into_sections(body)
    if not sections:
        sections = [_Section(label="other", body=_normalize_text(body))]

    chunks: list[Chunk] = []
    global_idx = 0

    for section in sections:
        text = section.body
        if not text.strip():
            continue

        tokens = estimate_tokens(text)
        label = section.label

        if tokens <= MAX_TOKENS:
            if tokens >= MIN_TOKENS:
                chunks.append(
                    Chunk(
                        chunk_id=f"{slug}__{label}__{global_idx}",
                        scheme_name=scheme_name,
                        scheme_category=scheme_category,
                        source_url=source_url,
                        snapshot_date=snapshot_date,
                        section=label,
                        text=text,
                        token_count=tokens,
                    )
                )
                global_idx += 1
        else:
            for sub_text in _split_long_section(text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{slug}__{label}__{global_idx}",
                        scheme_name=scheme_name,
                        scheme_category=scheme_category,
                        source_url=source_url,
                        snapshot_date=snapshot_date,
                        section=label,
                        text=sub_text,
                        token_count=estimate_tokens(sub_text),
                    )
                )
                global_idx += 1

    return chunks
