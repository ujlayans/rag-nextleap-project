"""Chunk dataclass — the unit of retrieval in the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A single citation-ready text chunk.

    Attributes
    ----------
    chunk_id:
        Deterministic ID: ``{scheme_slug}__{section}__{idx}``.
    scheme_name:
        Full scheme name as listed on Groww (e.g. ``"SBI Small Cap Fund Direct Growth"``).
    scheme_category:
        Category (e.g. ``"Small Cap"``).
    source_url:
        Exact Groww URL this chunk was extracted from.
    snapshot_date:
        ISO date the source page was fetched (``YYYY-MM-DD``).
    section:
        Canonical section label (``"fund_details"`` | ``"costs_tax"`` | ``"managers"`` | …).
    text:
        The chunk content. Guaranteed ≤ 512 tokens.
    token_count:
        Estimated token count of *text*.
    """

    chunk_id: str
    scheme_name: str
    scheme_category: str
    source_url: str
    snapshot_date: str
    section: str
    text: str
    token_count: int

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain dict ready for ``json.dumps``."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        """Reconstruct a ``Chunk`` from a dict (e.g. a JSONL line)."""
        return cls(
            chunk_id=d["chunk_id"],
            scheme_name=d["scheme_name"],
            scheme_category=d["scheme_category"],
            source_url=d["source_url"],
            snapshot_date=d["snapshot_date"],
            section=d["section"],
            text=d["text"],
            token_count=int(d["token_count"]),
        )
