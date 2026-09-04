# PRD: SBI Mutual Fund Facts Assistant (RAG Prototype)

**Product:** Groww-style FAQ chatbot for SBI Direct Growth schemes  
**Type:** Working RAG prototype (hobby / skills test)  
**Owner:** Product  
**Date:** 3 Sep 2026  
**Status:** Prototype v1

---

## 1. Problem

Investors looking at SBI schemes on Groww need **fast, citable facts** (expense ratio, SIP minimum, exit load, riskometer, lock-in). Today they scan long scheme pages. They also get **unsafe answers** from generic chatbots: buy/sell advice, return comparisons, and invented numbers.

We will ship a **facts-only RAG assistant** that answers from a **closed corpus of seven Groww scheme pages** and refuses anything else.

## 2. Who it is for

| User | Need |
|------|------|
| Curious investor | Look up a scheme fact without opening every page |
| Reviewer / evaluator | See RAG retrieve the right chunk, cite one URL, refuse advice |

Not for: portfolio construction, tax filing, account support, or live trading.

## 3. Goal

A user can ask a factual question about **one of seven SBI Direct Growth schemes** and get:

1. An answer of **≤3 sentences** grounded in retrieved text  
2. **One citation link** (the Groww scheme URL used)  
3. A line: **Last updated from sources: \<date\>**  
4. A clear **refuse** if the question is advice, returns comparison, PII, or out of corpus  

**Non-goal:** beat a full AMC chatbot. This is a small-corpus RAG demo.

## 4. In-scope schemes (corpus = these URLs only)

AMC: **SBI Mutual Fund**. Plan: **Direct Growth**. Source site: **groww.in**.

| # | Category | Scheme (as on Groww) | URL |
|---|----------|----------------------|-----|
| 1 | Small Cap | SBI Small Cap Fund Direct Growth | https://groww.in/mutual-funds/sbi-small-midcap-fund-direct-growth |
| 2 | Mid Cap | SBI Mid Cap Direct Plan Growth | https://groww.in/mutual-funds/sbi-mid-cap-direct-plan-growth |
| 3 | Large Cap | SBI Large Cap Direct Plan Growth | https://groww.in/mutual-funds/sbi-large-cap-direct-plan-growth |
| 4 | Nifty | SBI Nifty Next 50 Index Fund Direct Growth | https://groww.in/mutual-funds/sbi-nifty-next-50-index-fund-direct-growth |
| 5 | Gold | SBI Gold Direct Plan Growth | https://groww.in/mutual-funds/sbi-gold-fund-direct-growth |
| 6 | ELSS | SBI ELSS Tax Saver Fund Direct Growth | https://groww.in/mutual-funds/sbi-elss-tax-saver-fund-direct-growth |
| 7 | Flexi Cap | SBI Flexicap Fund Direct Growth | https://groww.in/mutual-funds/sbi-flexicap-fund-direct-growth |

No other URLs, blogs, screenshots, or AMC back-end data.

## 5. User stories

1. **As an investor**, I ask “Expense ratio of SBI Small Cap Direct Growth?” and see the published ratio plus the Groww link.  
2. **As an investor**, I ask “ELSS lock-in?” and see the 3-year lock-in from the ELSS page.  
3. **As an investor**, I ask “Should I buy the gold fund?” and get a polite facts-only refusal (no advice).  
4. **As an evaluator**, I see the UI welcome, three sample questions, and “Facts-only. No investment advice.”

## 6. Functional requirements

| ID | Requirement |
|----|-------------|
| F1 | Chat UI: welcome line, 3 example questions, disclaimer |
| F2 | Answer factual queries: expense ratio, min SIP / lumpsum, exit load, riskometer, category, benchmark, NAV date, AUM, fund managers, ELSS lock-in, tax notes **as printed on the page** |
| F3 | Every answer includes **exactly one** citation URL from the corpus |
| F4 | Answers ≤3 sentences + “Last updated from sources: …” |
| F5 | Refuse opinion / buy-sell / “best fund” / allocation advice |
| F6 | Do not **compute or compare** returns; if asked for performance, point to the scheme page (do not rank or calculate) |
| F7 | Reject PII (PAN, Aadhaar, account numbers, OTP, email, phone); do not store it |
| F8 | If the fact is not in retrieved chunks, say so; do not invent |
| F9 | Groww-like accent colour (`#00B386`); Streamlit |

## 7. Non-functional / constraints

- Public sources only; corpus frozen to the seven links above  
- No PII persistence  
- Lightweight embedding: `sentence-transformers/all-MiniLM-L6-v2`  
- Vector store: **ChromaDB**  
- Generator: **Mistral API** (key in env; extractive fallback if key missing)  
- Prototype quality: local run via README; not a production SLA  

## 8. RAG design (product view)

```
Question → PII / advice / returns-compare guards
        → Embed query (same MiniLM model)
        → Retrieve top-k chunks from Chroma
        → Prompt: facts-only, ≤3 sentences, one citation, no invention
        → Mistral (or extractive fallback)
        → UI: answer + citation + last-updated
```

**Ingestion:** snapshot of each Groww page → chunk by section (facts, objective, costs/tax, managers) → embed → persist Chroma.

## 9. Guardrails (must refuse)

| Type | Example | Behaviour |
|------|---------|-----------|
| Advice | Should I sell mid cap? | Polite refusal + educational citation (scheme or definition chunk) |
| Returns | Which fund performed best? | No comparison; send user to the scheme page |
| PII | My PAN is … | Refuse; do not echo or store |
| Out of corpus | How to download capital-gains statement? | Not in these 7 pages → say unsupported |
| Other AMC / scheme | HDFC small cap | Out of scope |

## 10. UI copy

- **Welcome:** “Ask factual questions about seven SBI Direct Growth schemes listed on Groww.”  
- **Examples:** expense ratio (small cap); ELSS lock-in; min SIP (gold).  
- **Disclaimer:** “Facts-only. No investment advice. Not an offer to buy or sell. Verify on Groww / SID before acting. We do not collect PAN, Aadhaar, account numbers, OTPs, email, or phone.”

## 11. Success criteria (prototype)

- Retrieval hits the correct scheme for named-fund questions  
- Citation URL matches the scheme used  
- Advice and PII paths never call the model with user secrets (PII stripped / blocked first)  
- Edge-case tests in `tests/test_edge_cases.py` pass  
- README lists setup, scope, and known limits  

## 12. Out of scope (v1)

Live Groww login, order placement, personal holdings, return calculators, multi-AMC, screenshots of app back-end, third-party blogs.

## 13. Known limits (call out in README)

- Groww HTML can change; prototype uses a **dated snapshot** of the seven pages  
- NAV/AUM/expense ratio can go stale; “last updated” is the snapshot date  
- “How to download capital-gains statement” is **not** on the seven URLs → honest miss  
- MiniLM + small corpus can still retrieve the wrong scheme if the user does not name the fund  
