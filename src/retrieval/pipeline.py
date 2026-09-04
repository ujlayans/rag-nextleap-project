"""Retrieval pipeline — orchestrates guards → retrieve → generate → format."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from src.config import TOP_K
from src.embedding.encoder import Embedder
from src.retrieval.generator import generate
from src.retrieval.guards import run_guards
from src.retrieval.retriever import RetrievedChunk, Retriever
from src.vector_store.store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    text: str
    citation: str | None
    last_updated: str | None
    refused: bool = False
    refusal_kind: str | None = None
    scheme_name: str | None = None
    retrieved: list[str] = field(default_factory=list)  # chunk ids
    source: str | None = None  # "mistral" | "extractive"


class RetrievalPipeline:
    """Single entry point for the whole retrieval chain (UI calls plane)."""

    def __init__(
        self,
        *,
        top_k: int = TOP_K,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.top_k = top_k
        self.retriever = Retriever(
            embedder=embedder, store=store, top_k=top_k
        )

    def run(self, question: str) -> Answer:
        """Process a user question end-to-end."""
        # 1. Guards
        guard = run_guards(question)
        if guard.blocked:
            return Answer(
                text=guard.message or "",
                citation=guard.citation,
                last_updated=None,
                refused=True,
                refusal_kind=guard.kind,
                scheme_name=guard.scheme_name,
            )

        # 2. Retrieve (scheme filter if resolved)
        chunks: list[RetrievedChunk] = self.retriever.retrieve(
            question,
            scheme_name=guard.scheme_name,
            top_k=self.top_k,
        )

        # 3. Explicitly enforce "fact must be in the chunks" (PRD F8)
        if not chunks:
            return Answer(
                text=(
                    "I don't have this information in my sources. "
                    "Ask a factual question about one of the seven SBI schemes."
                ),
                citation=None,
                last_updated=None,
                refused=False,
            )

        # 4. Generate + format
        result_text = generate(question, chunks)
        citation, last_updated, body = _split_format(result_text.text, chunks)

        return Answer(
            text=body,
            citation=citation or chunks[0].source_url,
            last_updated=last_updated or chunks[0].snapshot_date,
            refused=False,
            scheme_name=guard.scheme_name or chunks[0].scheme_name,
            retrieved=[c.chunk_id for c in chunks],
            source=result_text.provider,
        )


def _split_format(result: str, chunks: list[RetrievedChunk]):
    """Split a generated answer into (citation, last_updated, body).

    Extractive fallbacks already embed these; for LLM responses we parse the
    ``Source:`` and ``Last updated from sources:`` lines out.
    """
    citation = None
    last_updated = None
    lines = result.splitlines()
    kept: list[str] = []

    import re

    for ln in lines:
        s = ln.strip()
        m = re.match(r"^Source:\s*(\S+)$", s, re.IGNORECASE)
        if m and "groww.in" in m.group(1):
            citation = m.group(1)
            continue
        m = re.match(r"^Last updated from sources:\s*(.+)$", s, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            # Ignore the prompt's literal placeholder if the model echoed it back.
            if val and not re.match(r"^<.*>$", val):
                last_updated = val
            continue
        kept.append(ln)

    body = "\n".join(kept).strip()
    if body:
        # Strip any lingering inline source markers the model may have echoed,
        # e.g. "[Source: https://groww.in/... | Section: fund_details]".
        body = re.sub(
            r"\[\s*Source:\s*[^\]\n]*\s*\|\s*Section:[^\]\n]*\]",
            "",
            body,
            flags=re.IGNORECASE,
        )
        # Drop markdown link syntax but keep the visible URL as plain text.
        body = re.sub(r"\[([^\]]*)\]\((https?://[^)]+)\)", r"\1", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return citation, last_updated, body


# Convenience single-call wrapper
#
# ChromaDB's PersistentClient/SharedSystemClient is NOT safe to keep alive
# across the threads Streamlit uses for reruns (raises `AttributeError:
# bindings`). We therefore cache only the heavy, thread-safe-for-inference
# Embedder model and open a *fresh* VectorStore per call, closing it
# immediately afterward. Reruns are sequential, so this is safe and cheap.
_model_lock = threading.Lock()
_model_singleton: Embedder | None = None


def _get_embedder() -> Embedder:
    global _model_singleton
    if _model_singleton is None:
        with _model_lock:
            if _model_singleton is None:
                _model_singleton = Embedder()
    return _model_singleton


def run_pipeline(question: str) -> Answer:
    pipeline = RetrievalPipeline(embedder=_get_embedder())
    try:
        return pipeline.run(question)
    finally:
        pipeline.retriever.store.close()
