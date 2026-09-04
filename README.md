# SBI Facts Assistant (RAG prototype)

Facts-only chatbot for **seven SBI Mutual Fund Direct Growth schemes** as listed on [Groww](https://groww.in/).  
It retrieves from a **closed corpus** (those URLs only), cites **one** Groww link per answer, and **refuses** investment advice.

Product spec: [DOCS/PRD.md](DOCS/PRD.md)

## Scope

| Category | Scheme | Groww URL |
|----------|--------|-----------|
| Small Cap | SBI Small Cap Fund Direct Growth | https://groww.in/mutual-funds/sbi-small-midcap-fund-direct-growth |
| Mid Cap | SBI Mid Cap Direct Plan Growth | https://groww.in/mutual-funds/sbi-mid-cap-direct-plan-growth |
| Large Cap | SBI Large Cap Direct Plan Growth | https://groww.in/mutual-funds/sbi-large-cap-direct-plan-growth |
| Nifty | SBI Nifty Next 50 Index Fund Direct Growth | https://groww.in/mutual-funds/sbi-nifty-next-50-index-fund-direct-growth |
| Gold | SBI Gold Direct Plan Growth | https://groww.in/mutual-funds/sbi-gold-fund-direct-growth |
| ELSS | SBI ELSS Tax Saver Fund Direct Growth | https://groww.in/mutual-funds/sbi-elss-tax-saver-fund-direct-growth |
| Flexi Cap | SBI Flexicap Fund Direct Growth | https://groww.in/mutual-funds/sbi-flexicap-fund-direct-growth |

AMC: **SBI**. Sources: **public Groww scheme pages only** (snapshot in `data/corpus/schemes.json`). Full list: `data/sources.csv`.

## RAG pipeline

1. **Ingest** structured snapshot → section chunks  
2. **Embed** with `sentence-transformers/all-MiniLM-L6-v2`  
3. **Store** in ChromaDB (`chroma_db/`)  
4. **Retrieve** top-k chunks for the question  
5. **Generate** with Mistral API (extractive fallback if no key)  
6. **UI** Streamlit, Groww accent `#00B386`

Guards run **before** retrieval: PII, buy/sell advice, return comparison.

## Setup

Python 3.10+.

```bash
cd "RAG project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional: add MISTRAL_API_KEY
python -m src.ingest
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

Without `MISTRAL_API_KEY`, answers are **extractive** (top retrieved chunk), still cited.

## Disclaimer (UI)

See [DOCS/disclaimer.txt](DOCS/disclaimer.txt). Short line on screen: **Facts-only. No investment advice.**

## Sample Q&A

[DOCS/sample_qa.md](DOCS/sample_qa.md)

## Edge-case tests

```bash
source .venv/bin/activate
python -m pytest tests/test_edge_cases.py -q
```

## Known limits

- Corpus is a **dated snapshot** (see “Last updated from sources” on answers), not a live scrape of Groww HTML.  
- **How to download a capital-gains statement** is not on these seven pages → assistant should say it is out of corpus.  
- Does **not** compute or compare returns; points you to the Groww page.  
- Wrong scheme can be retrieved if you do not name the fund.  
- Not affiliated with Groww or SBI Mutual Fund. Not for production or regulated advice.

## Deliverables

| Item | Path |
|------|------|
| PRD | `DOCS/PRD.md` |
| Prototype | `app.py` |
| Source list | `data/sources.csv` |
| Sample Q&A | `DOCS/sample_qa.md` |
| Disclaimer | `DOCS/disclaimer.txt` |
| This README | `README.md` |
