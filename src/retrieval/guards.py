"""Product guard pipeline — runs before any embedding/retrieval.

Detection set (PRD §9):
  - PII           → refuse, never echo/store
  - Advice        → refuse buy/sell/alloc allocation
  - Returns       → refuse comparison/calculation
  - Out of corpus → say unsupported (topic not on any of the 7 pages)
  - Other AMC     → only SBI Direct Growth schemes in scope
Scheme resolution is also performed here (named-fund queries set a metadata
filter for retrieval).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.data_loading.fetcher import SCHEME_URLS

# ---------------------------------------------------------------------------
# Closed corpus of the 7 in-scope schemes (PRD §4)
# ---------------------------------------------------------------------------

# scheme search key -> {name, url}. Fuzzy matching against these keys.
_SCHEME_KEYS: dict[str, str] = {
    "small cap": "https://groww.in/mutual-funds/sbi-small-midcap-fund-direct-growth",
    "small": "https://groww.in/mutual-funds/sbi-small-midcap-fund-direct-growth",
    "mid cap": "https://groww.in/mutual-funds/sbi-mid-cap-direct-plan-growth",
    "mid": "https://groww.in/mutual-funds/sbi-mid-cap-direct-plan-growth",
    "large cap": "https://groww.in/mutual-funds/sbi-large-cap-direct-plan-growth",
    "large": "https://groww.in/mutual-funds/sbi-large-cap-direct-plan-growth",
    "nifty next 50": "https://groww.in/mutual-funds/sbi-nifty-next-50-index-fund-direct-growth",
    "nifty": "https://groww.in/mutual-funds/sbi-nifty-next-50-index-fund-direct-growth",
    "next 50": "https://groww.in/mutual-funds/sbi-nifty-next-50-index-fund-direct-growth",
    "gold": "https://groww.in/mutual-funds/sbi-gold-fund-direct-growth",
    "elss": "https://groww.in/mutual-funds/sbi-elss-tax-saver-fund-direct-growth",
    "tax saver": "https://groww.in/mutual-funds/sbi-elss-tax-saver-fund-direct-growth",
    "flexicap": "https://groww.in/mutual-funds/sbi-flexicap-fund-direct-growth",
    "flexi cap": "https://groww.in/mutual-funds/sbi-flexicap-fund-direct-growth",
    "flexi": "https://groww.in/mutual-funds/sbi-flexicap-fund-direct-growth",
}

# Other / foreign AMCs we should refuse outright (PRD §9 "Other AMC")
_OTHER_AMC_PATTERNS = [
    r"\b(hdfc|icici|axis|kotak|nippon|sundaram|invesco|bandhan|uti|tata|franklin|hdfc|bajaj|quant|parag parikh|mirae|hsbc)\b",
]


@dataclass
class GuardResult:
    blocked: bool
    kind: str | None
    message: str | None
    citation: str | None = None
    # Set by the scheme-resolve step for named-fund queries.
    scheme_name: str | None = None
    scheme_url: str | None = None


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "PAN"),
    (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "Aadhaar-like number"),
    (r"\b\d{12}\b", "12-digit ID"),
    (r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "email"),
    (r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b", "phone"),
    (r"\b(?:otp|one[-\s]?time\s?password)\b.{0,20}\b\d{4,8}\b", "OTP"),
    (r"\b(?:account|a/c|folio)(?:\s*(?:no|number|#))?\s*[:\-]?\s*\d{6,}\b", "account/folio"),
]


def _check_pii(q: str) -> GuardResult:
    for pattern, label in _PII_PATTERNS:
        if re.search(pattern, q, flags=re.IGNORECASE):
            return GuardResult(
                blocked=True,
                kind="pii",
                message=(
                    "I cannot accept or store personal identifiers (PAN, Aadhaar, account numbers, "
                    "OTPs, email, or phone). Ask a scheme fact without sharing those details."
                ),
            )
    return GuardResult(blocked=False, kind=None, message=None)


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------

_ADVICE_PATTERNS = [
    r"\bshould i (buy|sell|invest|switch|redeem|hold|exit)\b",
    r"\b(buy|sell) (or )?not\b",
    r"\bis (it|this) (a )?good (time|fund|investment)\b",
    r"\bbest (fund|sip|scheme|option)\b",
    r"\brecommend(ed|ation)?\b",
    r"\bwhich (fund|one) (should|shall) i\b",
    r"\bportfolio\b",
    r"\bhow much (should|to) (i )?invest\b",
    r"\ballocat(e|ion)\b",
    r"\btips? for investing\b",
    r"\bguaranteed returns?\b",
]


def _check_advice(q: str) -> GuardResult:
    if any(re.search(p, q, re.IGNORECASE) for p in _ADVICE_PATTERNS):
        return GuardResult(
            blocked=True,
            kind="advice",
            message=(
                "I cannot give buy/sell or portfolio advice. I only repeat published scheme facts "
                "(expense ratio, SIP minimum, exit load, lock-in, riskometer). For suitability, use "
                "the scheme page and SID, or a SEBI-registered advisor."
            ),
        )
    return GuardResult(blocked=False, kind=None, message=None)


# ---------------------------------------------------------------------------
# Returns comparison
# ---------------------------------------------------------------------------

_RETURNS_PATTERNS = [
    r"\b(best|worst|highest|lowest) (return|perform)",
    r"\bcompar(e|ison)\b.{0,40}\b(return|performance|cagr)s?\b",
    r"\bwhich (fund|scheme).{0,30}(better|outperform|beat)\b",
    r"\b(better|outperform|beat)\b.{0,30}\b(?:or|and)\b",
    r"\b(best|worst) performing\b",
    r"\b(1|3|5|10)[\s-]?y(ear|r)? (return|cagr|performance)s?\b",
    r"\bannualised returns?\b",
    r"\bwhat (were|are|was|is) (the )?returns?\b",
    r"\bhow (much|did).{0,20}(return|gain|grow)\b",
    r"\bxirr\b",
    r"\bcagr\b",
]


def _check_returns(q: str) -> GuardResult:
    if any(re.search(p, q, re.IGNORECASE) for p in _RETURNS_PATTERNS):
        return GuardResult(
            blocked=True,
            kind="returns",
            message=(
                "I do not compute or compare returns. See the official scheme page / factsheet on "
                "Groww for published performance. Past performance is not a guarantee of future results."
            ),
        )
    return GuardResult(blocked=False, kind=None, message=None)


# ---------------------------------------------------------------------------
# Other AMC / out of corpus
# ---------------------------------------------------------------------------

def _check_other_amc(q: str) -> GuardResult:
    lowered = q.lower()
    # Only flag a foreign AMC if the query is actually about a fund, to avoid
    # over-flagging e.g. generic words.
    is_fund_question = any(
        kw in lowered
        for kw in ("fund", "scheme", " sip", "nav", "expense ratio", "exit load", "return", "invest")
    )
    if is_fund_question:
        for pat in _OTHER_AMC_PATTERNS:
            if re.search(pat, lowered):
                return GuardResult(
                    blocked=True,
                    kind="other_amc",
                    message=(
                        "I can only answer questions about the seven SBI Direct Growth schemes "
                        "listed on Groww."
                    ),
                )
    return GuardResult(blocked=False, kind=None, message=None)


def _check_out_of_corpus(q: str, resolved_scheme: str | None) -> GuardResult:
    # If a scheme was resolved, we assume the query is about a supported fund.
    if resolved_scheme:
        return GuardResult(blocked=False, kind=None, message=None)

    # Otherwise the question might still be about a generic supported fact
    # (e.g. "what is expense ratio?"). If it matches no supported topic at all,
    # call it out of corpus.
    supported_topic = re.search(
        r"(expense ratio|sip|lumpsum|exit load|lock[- ]?in|riskometer|nav|aum|fund size|"
        r"fund manager|benchmark|tax|category|rating|holdings|objective|minimum)",
        q,
        re.IGNORECASE,
    )
    if not supported_topic:
        return GuardResult(
            blocked=True,
            kind="out_of_corpus",
            message=(
                "That question is outside the seven scheme pages I cover. "
                "Ask about facts such as expense ratio, SIP minimum, exit load, lock-in, riskometer, "
                "NAV, AUM, fund manager, or benchmark for one of the seven SBI schemes."
            ),
        )
    return GuardResult(blocked=False, kind=None, message=None)


# ---------------------------------------------------------------------------
# Scheme resolution
# ---------------------------------------------------------------------------

def _resolve_scheme(q: str) -> tuple[str | None, str | None]:
    """Return ``(scheme_name_or_none, url_or_none)`` if exactly one scheme is named."""
    lowered = q.lower()
    matches: set[str] = set()
    for key, url in _SCHEME_KEYS.items():
        if key in lowered:
            matches.add(url)
    if len(matches) == 1:
        url = next(iter(matches))
        # reverse-lookup a canonical name from SCHEME_URLS
        name = _name_for_url(url)
        return name, url
    return None, None


def _name_for_url(url: str) -> str:
    from src.data_loading.parser import _SCHEME_NAME_MAP  # avoid circular import at top-level

    for slug, u in SCHEME_URLS.items():
        if u == url:
            return _SCHEME_NAME_MAP.get(slug, slug)
    return ""


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_guards(question: str) -> GuardResult:
    """Run the full guard pipeline. Returns a non-blocked result on success,
    carrying an optional resolved ``scheme_name`` / ``scheme_url``."""
    q = (question or "").strip()

    if not q:
        return GuardResult(
            blocked=True,
            kind="empty",
            message="Please type a factual question about one of the seven SBI schemes in scope.",
        )

    # Order matters: PII first (never process secrets), then AMC, advice, returns.
    for fn in (_check_pii, _check_other_amc, _check_advice, _check_returns):
        result = fn(q)
        if result.blocked:
            return result

    # Scheme resolution drives both the corpus check and the metadata filter.
    scheme_name, scheme_url = _resolve_scheme(q)
    corpus = _check_out_of_corpus(q, scheme_name)
    if corpus.blocked:
        return corpus

    return GuardResult(
        blocked=False,
        kind=None,
        message=None,
        scheme_name=scheme_name,
        scheme_url=scheme_url,
    )
