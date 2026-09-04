"""Edge cases for the facts-only assistant (no live LLM required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.guards import check_advice, check_pii, check_returns, run_guards
from src.ingest import build_chunks, load_corpus


def test_empty_question_blocked():
    r = run_guards("   ")
    assert r.blocked and r.kind == "empty"


def test_pan_rejected_and_not_echoed():
    r = check_pii("My PAN is ABCDE1234F what is expense ratio")
    assert r.blocked and r.kind == "pii"
    assert "ABCDE1234F" not in (r.message or "")


def test_phone_rejected():
    r = check_pii("Call me on 9876543210 about ELSS")
    assert r.blocked and r.kind == "pii"


def test_email_rejected():
    r = check_pii("mail results to user@example.com")
    assert r.blocked and r.kind == "pii"


def test_buy_advice_refused():
    r = check_advice("Should I buy SBI Small Cap Fund?")
    assert r.blocked and r.kind == "advice"
    assert "cannot give" in (r.message or "").lower() or "advice" in (r.message or "").lower()


def test_best_fund_refused():
    r = check_advice("Which is the best fund for my portfolio?")
    assert r.blocked


def test_returns_comparison_refused():
    r = check_returns("Which fund has the best 3 year return?")
    assert r.blocked and r.kind == "returns"


def test_cagr_refused():
    r = check_returns("What is the CAGR of gold fund?")
    assert r.blocked


def test_factual_expense_ratio_not_blocked():
    r = run_guards("What is the expense ratio of SBI Small Cap Fund Direct Growth?")
    assert not r.blocked


def test_elss_lockin_not_blocked():
    r = run_guards("What is the ELSS lock-in for SBI ELSS Tax Saver Fund?")
    assert not r.blocked


def test_min_sip_not_blocked():
    r = run_guards("What is the minimum SIP for SBI Gold Fund Direct Growth?")
    assert not r.blocked


def test_corpus_has_seven_schemes():
    corpus = load_corpus()
    assert len(corpus["schemes"]) == 7
    urls = {s["url"] for s in corpus["schemes"]}
    assert all(u.startswith("https://groww.in/mutual-funds/") for u in urls)


def test_chunks_carry_citation_url():
    chunks = build_chunks(load_corpus())
    assert len(chunks) >= 21
    assert all(c["url"].startswith("https://groww.in/") for c in chunks)


def test_elss_lockin_in_corpus():
    text = " ".join(c["text"] for c in build_chunks(load_corpus()) if c["scheme"].startswith("SBI ELSS"))
    assert "3-year" in text or "3Y" in text
