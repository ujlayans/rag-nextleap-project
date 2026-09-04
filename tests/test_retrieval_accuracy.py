"""Tests for the Phase 5 retrieval chain (src.retrieval.retriever + pipeline).

These exercise real embedding inference (MiniLM) against the persisted ChromaDB
store in ``data/vectordb``. They DO NOT call the Mistral API — answers come from
the extractive fallback, so they are deterministic and offline.

Datasets are tiny in-memory and fast; annotations mark them integration-level.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.guards import run_guards
from src.retrieval.pipeline import RetrievalPipeline, run_pipeline
from src.retrieval.retriever import Retriever


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever(top_k=4)


# ---------------------------------------------------------------------------
# Guard → pipeline refusal paths (no store needed)
# ---------------------------------------------------------------------------

def test_pipeline_refuses_advice():
    a = run_pipeline("Should I buy SBI Small Cap?")
    assert a.refused and a.refusal_kind == "advice"
    assert a.citation is None


def test_pipeline_refuses_returns():
    a = run_pipeline("What was the 1-year return of SBI Gold?")
    assert a.refused and a.refusal_kind == "returns"


# ---------------------------------------------------------------------------
# Retrieval accuracy — scheme filter correctness (mini integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_scheme_filter_restricts_to_one_scheme(retriever):
    """A --scheme filtered query must return only chunks of that scheme."""
    hits = retriever.retrieve(
        "expense ratio",
        scheme_name="SBI Small Cap Fund Direct Growth",
        top_k=8,
    )
    assert hits, "expected at least one hit"
    schemes = {h.scheme_name for h in hits}
    assert schemes == {"SBI Small Cap Fund Direct Growth"}


@pytest.mark.integration
def test_scheme_filter_returns_gold_for_gold_query(retriever):
    hits = retriever.retrieve(
        "gold fund NAV",
        scheme_name="SBI Gold Direct Plan Growth",
        top_k=4,
    )
    assert hits
    assert all(h.scheme_name == "SBI Gold Direct Plan Growth" for h in hits)


@pytest.mark.integration
def test_unfiltered_query_returns_multiple_schemes(retriever):
    hits = retriever.retrieve("what is the expense ratio", top_k=6)
    assert hits
    assert len({h.scheme_name for h in hits}) >= 1


# ---------------------------------------------------------------------------
# Pipeline output formatting (extractive fallback, offline)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_pipeline_answer_has_citation_and_update():
    a = run_pipeline("What is the lock-in period for SBI ELSS Tax Saver Fund?")
    assert not a.refused
    assert a.citation and "groww.in" in a.citation
    assert a.last_updated
    assert a.text


@pytest.mark.integration
def test_pipeline_resolves_scheme_name():
    a = run_pipeline("What is the expense ratio of SBI Small Cap Fund Direct Growth?")
    assert a.scheme_name == "SBI Small Cap Fund Direct Growth"
