"""Product guards: PII, advice, and returns comparison — before retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

PII_PATTERNS = [
    (r"\b[A-Z]{5}\d{4}[A-Z]\b", "PAN"),
    (r"\b\d{4}\s?\d{4}\s?\d{4}\b", "Aadhaar-like number"),
    (r"\b\d{12}\b", "12-digit ID"),
    (r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "email"),
    (r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b", "phone"),
    (r"\b(?:otp|one[-\s]?time\s?password)\b.{0,20}\b\d{4,8}\b", "OTP"),
    (r"\b(?:account|a/c|folio)(?:\s*(?:no|number|#))?\s*[:\-]?\s*\d{6,}\b", "account/folio"),
]

ADVICE_PATTERNS = [
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

RETURNS_COMPARE_PATTERNS = [
    r"\b(best|worst|highest|lowest) (return|perform)",
    r"\bcompar(e|ison)\b.{0,40}\b(return|performance|cagr)\b",
    r"\bwhich (fund|scheme).{0,30}(better|outperform|beat)\b",
    r"\b(3|5|10)[\s-]?y(ear)? (return|cagr|performance)\b",
    r"\bannualised returns?\b",
    r"\bwhat (were|are|is) (the )?returns?\b",
    r"\bhow (much|did).{0,20}(return|gain|grow)\b",
    r"\bxirr\b",
    r"\bcagr\b",
]


@dataclass
class GuardResult:
    blocked: bool
    kind: str | None
    message: str | None
    citation: str | None = None


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def check_pii(question: str) -> GuardResult:
    for pattern, label in PII_PATTERNS:
        flags = re.IGNORECASE if "email" in label.lower() or "otp" in label.lower() or "account" in label.lower() else 0
        if label == "PAN":
            flags = 0
        if re.search(pattern, question, flags=flags | re.IGNORECASE):
            return GuardResult(
                blocked=True,
                kind="pii",
                message=(
                    "I cannot accept or store personal identifiers (PAN, Aadhaar, account numbers, OTPs, email, or phone). "
                    "Ask a scheme fact without sharing those details."
                ),
                citation="https://groww.in/mutual-funds/sbi-elss-tax-saver-fund-direct-growth",
            )
    return GuardResult(blocked=False, kind=None, message=None)


def check_advice(question: str) -> GuardResult:
    if _match_any(question, ADVICE_PATTERNS):
        return GuardResult(
            blocked=True,
            kind="advice",
            message=(
                "I cannot give buy/sell or portfolio advice. "
                "I only repeat published scheme facts (expense ratio, SIP minimum, exit load, lock-in, riskometer). "
                "For suitability, use the scheme page and SID, or a SEBI-registered advisor."
            ),
            citation="https://groww.in/mutual-funds/sbi-elss-tax-saver-fund-direct-growth",
        )
    return GuardResult(blocked=False, kind=None, message=None)


def check_returns(question: str) -> GuardResult:
    if _match_any(question, RETURNS_COMPARE_PATTERNS):
        return GuardResult(
            blocked=True,
            kind="returns",
            message=(
                "I do not compute or compare returns. "
                "See the official scheme page / factsheet on Groww for published performance. "
                "Past performance is not a guarantee of future results."
            ),
            citation="https://groww.in/mutual-funds/sbi-small-midcap-fund-direct-growth",
        )
    return GuardResult(blocked=False, kind=None, message=None)


def run_guards(question: str) -> GuardResult:
    q = (question or "").strip()
    if not q:
        return GuardResult(
            blocked=True,
            kind="empty",
            message="Please type a factual question about one of the seven SBI schemes in scope.",
            citation="https://groww.in/mutual-funds/sbi-small-midcap-fund-direct-growth",
        )
    for fn in (check_pii, check_advice, check_returns):
        result = fn(q)
        if result.blocked:
            return result
    return GuardResult(blocked=False, kind=None, message=None)
