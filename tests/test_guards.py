"""Tests for the Phase 5 guard pipeline (src.retrieval.guards).

Exercises the guard order (PII → other-AMC → advice → returns → corpus) plus
scheme resolution. These are pure string-matching tests — no embedding, no store,
no live LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval.guards import (
    GuardResult,
    _check_advice,
    _check_other_amc,
    _check_out_of_corpus,
    _check_pii,
    _check_returns,
    _resolve_scheme,
    run_guards,
)


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------

def test_pan_rejected():
    r = _check_pii("My PAN is ABCDE1234F")
    assert r.blocked and r.kind == "pii"
    assert "ABCDE1234F" not in (r.message or "")


def test_phone_rejected():
    r = _check_pii("Call 9876543210 for facts")
    assert r.blocked and r.kind == "pii"


def test_email_rejected():
    r = _check_pii("mail me at user@example.com")
    assert r.blocked and r.kind == "pii"


def test_plain_fact_not_pii():
    r = _check_pii("What is the expense ratio?")
    assert not r.blocked


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------

def test_buy_advice_refused():
    r = _check_advice("Should I buy SBI Small Cap?")
    assert r.blocked and r.kind == "advice"


def test_portfolio_refused():
    r = _check_advice("Help me pick for my portfolio")
    assert r.blocked and r.kind == "advice"


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def test_year_return_refused():
    r = _check_returns("What was the 1-year return of SBI Gold?")
    assert r.blocked and r.kind == "returns"


def test_returns_plural_refused():
    r = _check_returns("What were the 3 year returns?")
    assert r.blocked and r.kind == "returns"


def test_compare_returns_refused():
    r = _check_returns("Compare SBI Small Cap and SBI Mid Cap returns")
    assert r.blocked and r.kind == "returns"


def test_better_or_refused():
    r = _check_returns("Which is better SBI Flexicap or SBI Large Cap?")
    assert r.blocked and r.kind == "returns"


# ---------------------------------------------------------------------------
# Other AMC
# ---------------------------------------------------------------------------

def test_other_amc_refused():
    r = _check_other_amc("Is Nippon India a good fund?")
    assert r.blocked and r.kind == "other_amc"


def test_generic_word_not_amc():
    # "gold" here is not a foreign AMC — must not flag.
    r = _check_other_amc("What is the NAV of gold?")
    assert not r.blocked


# ---------------------------------------------------------------------------
# Out of corpus
# ---------------------------------------------------------------------------

def test_offtopic_out_of_corpus():
    r = _check_out_of_corpus("What is the capital of France?", resolved_scheme=None)
    assert r.blocked and r.kind == "out_of_corpus"


def test_supported_topic_passes():
    r = _check_out_of_corpus("What is the expense ratio?", resolved_scheme=None)
    assert not r.blocked


def test_resolved_scheme_skips_corpus():
    r = _check_out_of_corpus("anything at all", resolved_scheme="SBI Gold")
    assert not r.blocked


# ---------------------------------------------------------------------------
# Scheme resolution
# ---------------------------------------------------------------------------

def test_resolve_single_scheme():
    name, url = _resolve_scheme("Tell me about SBI Gold")
    assert name == "SBI Gold Direct Plan Growth"
    assert "sbi-gold-fund-direct-growth" in url


def test_resolve_ambiguous_two_schemes():
    name, url = _resolve_scheme("Compare SBI Small Cap and SBI Mid Cap")
    assert name is None and url is None


def test_resolve_none():
    name, url = _resolve_scheme("What is the weather?")
    assert name is None and url is None


# ---------------------------------------------------------------------------
# Full pipeline: guard ordering
# ---------------------------------------------------------------------------

def test_empty_blocked():
    r = run_guards("   ")
    assert r.blocked and r.kind == "empty"


def test_pii_beats_advice_order():
    # PII must fire first even when the phrasing also smells like advice.
    r = run_guards("Should I buy, my PAN is ABCDE1234F")
    assert r.blocked and r.kind == "pii"


def test_factual_pass_with_resolved_scheme():
    r = run_guards("What is the SIP minimum for SBI ELSS Tax Saver Fund?")
    assert not r.blocked
    assert r.scheme_name == "SBI ELSS Tax Saver Fund Direct Growth"


def test_factual_generic_pass():
    r = run_guards("What is the expense ratio?")
    assert not r.blocked
