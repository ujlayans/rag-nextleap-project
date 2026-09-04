# Architecture: SBI Mutual Fund Facts Assistant (RAG Prototype)

> Derived strictly from the PRD. No scope creep.

---

## High-Level Overview

```
┌──────────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐
│  Data Loader │───▶│ Chunker  │───▶│ Embedder │───▶│ VectorStore│───▶│ Retriever  │───▶│   LLM    │
│  (crawl/     │    │          │    │          │    │ (ChromaDB) │    │ + Guards   │    │ (Mistral)│
│   snapshot)  │    │          │    │          │    │            │    │            │    │          │
└──────────────┘    └──────────┘    └──────────┘    └────────────┘    └────────────┘    └──────────┘
                                                                                          │
                                                                                          ▼
                                                                                    ┌──────────┐
                                                                                    │ Streamlit│
                                                                                    │   UI     │
                                                                                    └──────────┘
```

**Hard constraints (from PRD §7):**
- Corpus: 7 Groww URLs only
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: ChromaDB
- Generator: Mistral API (extractive fallback if key missing)
- UI: Streamlit

---

## Phase 1 — Data Loading

### Purpose
Fetch a dated HTML snapshot of each of the 7 Groww scheme pages and persist them as local files.

### Design

```
URLs (PRD §4) ──▶ httpx / requests ──▶ raw HTML ──▶ BeautifulSoup / trafilatura ──▶ clean markdown/text ──▶ data/snapshots/{scheme_slug}.md
```

### Components

| Component | Detail |
|-----------|--------|
| URL list | 7 hardcoded URLs from PRD §4. No discovery, no dynamic URL injection. |
| HTTP client | `httpx` with timeout + retry. Single-threaded is fine for 7 pages. |
| HTML extractor | `trafilatura` (or `BeautifulSoup` with `get_text`) to strip nav/footer/ads and retain main content only. |
| Output format | One Markdown file per scheme under `data/snapshots/`. Filename = slugified scheme name. |
| Metadata | Each snapshot file gets a YAML front-matter header: `scheme_name`, `category`, `source_url`, `snapshot_date`. |
| Dedup | If snapshot already exists for today, skip fetch. `--force` flag to re-fetch. |

### Guardrails
- Only fetch URLs from the approved list (PRD §4). Reject anything else.
- Store no PII. Pages are public Groww listings.

---

## Phase 2 — Chunking

### Purpose
Split each snapshot into semantically meaningful, citation-ready chunks.

### Design

```
snapshot.md ──▶ Section splitter ──▶ sub-chunks (max 512 tokens) ──▶ Chunk objects
```

### Chunk Object Schema

```python
@dataclass
class Chunk:
    chunk_id: str          # {scheme_slug}__{section}__{idx}
    scheme_name: str       # e.g. "SBI Small Cap Fund Direct Growth"
    scheme_category: str   # "Small Cap"
    source_url: str        # exact Groww URL from PRD §4
    snapshot_date: str     # YYYY-MM-DD
    section: str           # "overview" | "objective" | "fund_details" | "costs_tax" | "managers" | "riskometer" | "other"
    text: str              # ≤512 tokens
    token_count: int
```

### Chunking Rules

| Rule | Detail |
|------|--------|
| Primary split | By Groww page section headings (Overview, Objective, Fund Details, Costs & Tax, Fund Manager, Riskometer, etc.) |
| Secondary split | If a section exceeds 512 tokens, split on paragraph boundaries. Overlap: 50 tokens. |
| Minimum chunk | ≥ 50 tokens (discard empty / boilerplate fragments). |
| Citationability | Every chunk must carry `source_url` and `scheme_name` so the generator can cite it (PRD §6 F3). |

### Storage
- Persist chunk list as `data/chunks/chunks.jsonl` for inspection / debugging.
- Each line is a serialized `Chunk`.

---

## Phase 3 — Embedding

### Purpose
Convert chunks into dense vectors using a local sentence-transformer.

### Design

```
chunks ──▶ sentence-transformers/all-MiniLM-L6-v2 ──▶ vectors (384-dim)
```

### Components

| Component | Detail |
|-----------|--------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` (PRD §7). Loaded once at startup. |
| Batch size | 64 (tune for memory). |
| Normalization | L2-normalize embeddings for cosine similarity. |
| Query embedding | Same model, called at query time on the user question. |

### Guardrails
- Embedding model is frozen. No fine-tuning in v1.
- All chunks and queries use the same model — never mix.

---

## Phase 4 — Vector Store (ChromaDB)

### Purpose
Persist chunk vectors + metadata and serve fast similarity search.

### Design

```
                    ┌─────────────────────────────┐
                    │         ChromaDB             │
                    │  Collection: "sbi_schemes"   │
                    │                              │
                    │  id        = chunk_id         │
                    │  embedding = 384-dim vector   │
                    │  document  = chunk.text        │
                    │  metadata  = {                 │
                    │    scheme_name,                │
                    │    scheme_category,            │
                    │    source_url,                 │
                    │    snapshot_date,              │
                    │    section                     │
                    │  }                             │
                    └─────────────────────────────┘
```

### Operations

| Operation | Detail |
|-----------|--------|
| Ingest | Upsert chunks into collection. Idempotent (chunk_id is deterministic). |
| Query | `collection.query(query_embedding, n_results=k, where={...})`. Default `k=4`. |
| Filter | Optional metadata filter: `where={"scheme_name": "SBI Small Cap Fund Direct Growth"}` for named-fund queries. |
| Persistence | ChromaDB persisted to `data/chroma/` directory. Re-ingest if snapshot date changes. |
| Reset | `--rebuild` flag drops and recreates the collection from scratch. |

### Guardrails
- Only 7 schemes exist. Any `scheme_name` outside the known list is rejected at query time.

---

## Phase 5 — Retrieval Logic

### Purpose
Take a user question, run guards, retrieve relevant chunks, generate an answer, and return it with citation.

### Design

```
User Question
    │
    ▼
┌───────────────────┐
│  Guard Pipeline    │
│  ├─ PII detector   │──▶ reject + "We don't collect PII"
│  ├─ Advice filter  │──▶ refuse + educational citation
│  ├─ Returns check  │──▶ "Please check the scheme page"
│  ├─ Corpus check   │──▶ "Not in supported corpus"
│  └─ Scheme resolve │──▶ extract scheme name (if mentioned)
└───────┬───────────┘
        │ (pass)
        ▼
┌───────────────────┐
│  Embed query       │──▶ MiniLM → 384-dim vector
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  ChromaDB query    │──▶ top-k chunks (k=4, with optional scheme filter)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Prompt assembly   │──▶ System: facts-only, ≤3 sentences, one citation, no invention
│                    │    Context: retrieved chunks (text + source_url)
│                    │    User: original question
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Generator         │
│  ├─ Mistral API    │──▶ primary (if MISTRAL_API_KEY set)
│  └─ Extractive     │──▶ fallback (pick best chunk, return verbatim)
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  Response format   │
│  ├─ Answer (≤3 s) │
│  ├─ Citation URL   │
│  └─ "Last updated  │
│       from sources: │
│       {date}"       │
└───────────────────┘
```

### Guard Details (PRD §9)

| Guard | Detection method | Response |
|-------|-----------------|----------|
| PII | Regex for PAN ( `[A-Z]{5}[0-9]{4}[A-Z]` ), Aadhaar (12-digit), phone, email, OTP | Block immediately. Do not embed or query. |
| Advice | Keyword/pattern: "should I buy/sell", "best fund", "recommend", "advice" | Polite refusal + link to the relevant scheme page (if guessable) or corpus-level citation. |
| Returns comparison | "which fund performed", "best returns", "compare", "CAGR" | Refuse comparison. Link to individual scheme page. |
| Out of corpus | Scheme name not in the 7-list, or topic not on any page | "This information is not available in the supported corpus." |
| Other AMC | Any non-SBI scheme name detected | "I can only answer questions about SBI Direct Growth schemes listed on Groww." |

### Prompt Template

```
You are a facts-only assistant for SBI Mutual Fund schemes on Groww.

RULES:
1. Answer in ≤3 sentences using ONLY the provided context.
2. Include exactly one citation URL from the context.
3. End with: "Last updated from sources: <snapshot_date>"
4. If the context does not contain the answer, say: "I don't have this information in my sources."
5. Never give buy/sell advice. Never compare returns. Never invent numbers.

Context:
{chunks_with_urls}

Question: {user_question}
```

### Fallback (no Mistral key)
Extract the most relevant chunk by similarity score and return its text verbatim, prefixed with the citation URL.

---

## Phase 6 — Retrieval Testing

### Purpose
Validate end-to-end correctness per PRD §11 success criteria.

### Test Categories

#### 6a. Retrieval Accuracy Tests

| Test | Input | Expected |
|------|-------|----------|
| Named scheme query | "Expense ratio of SBI Small Cap Fund Direct Growth?" | Chunks from SBI Small Cap scheme; citation URL = correct Groww link. |
| ELSS lock-in | "What is the ELSS lock-in period?" | Chunk from ELSS scheme; answer mentions 3 years. |
| Min SIP (Gold) | "Minimum SIP for SBI Gold Direct Plan Growth?" | Chunk from Gold scheme; answer contains published minimum. |
| Unnamed scheme ambiguity | "What is the expense ratio?" (no scheme named) | Retrieves from multiple schemes; response asks user to specify OR returns top-k with caveat. |

#### 6b. Guard Tests

| Test | Input | Expected |
|------|-------|----------|
| PII rejection | "My PAN is ABCDE1234F, what's the expense ratio?" | Refuse. No query made to vector store. |
| Advice refusal | "Should I buy SBI Small Cap?" | Polite refusal. No investment advice given. |
| Returns comparison | "Which SBI fund has the best returns?" | Refuse comparison. Link to scheme pages. |
| Out of corpus | "What is the expense ratio of HDFC Small Cap?" | "Not in supported corpus" or similar. |
| Other AMC | "Tell me about ICICI Prudential" | Refuse. Only SBI schemes. |

#### 6c. Output Format Tests

| Test | Assertion |
|------|-----------|
| Citation present | Every answer contains exactly one `https://groww.in/...` URL. |
| Length | Answer ≤ 3 sentences. |
| Last-updated line | Response ends with "Last updated from sources: {date}". |
| No hallucination | Answer text matches content in retrieved chunks. |

#### 6d. Edge Case Tests (`tests/test_edge_cases.py`)

| Case | Detail |
|------|--------|
| Empty query | Return polite prompt to ask a question. |
| Very long query | Truncate or handle gracefully. |
| Non-English query | Refuse politely in English. |
| HTML change / stale snapshot | If snapshot date is old, still answer but flag staleness. |
| Double scheme name | "SBI Small Cap and SBI Gold" — handle or ask user to pick one. |

### Test Infrastructure

```
tests/
├── test_retrieval_accuracy.py    # Phase 6a
├── test_guards.py                # Phase 6b
├── test_output_format.py         # Phase 6c
└── test_edge_cases.py            # Phase 6d
```

- Framework: `pytest`
- Run: `pytest tests/ -v`
- No external API calls in tests — mock Mistral; use local ChromaDB.

---

## Directory Structure (v1)

> Clear separation: **data/** holds all generated artifacts, **src/** holds all code, **tests/** holds all tests. Nothing else lives at the root.

```
.
├── DOCS/
│   ├── PRD.md
│   └── architecture.md
│
│
├── data/                              ── all generated / persisted artifacts
│   ├── raw/                           Phase 1 — raw HTML snapshots from Groww
│   │   ├── sbi-small-cap-fund.html
│   │   ├── sbi-mid-cap-fund.html
│   │   ├── sbi-large-cap-fund.html
│   │   ├── sbi-nifty-next-50.html
│   │   ├── sbi-gold-fund.html
│   │   ├── sbi-elss-tax-saver.html
│   │   └── sbi-flexicap-fund.html
│   │
│   ├── chunks/                        Phase 2 — chunked JSONL + manifest
│   │   ├── chunks.jsonl               one line per Chunk object
│   │   └── manifest.json              scheme → chunk count, section map, snapshot date
│   │
│   ├── embeddings/                    Phase 3 — persisted embedding vectors
│   │   ├── vectors.npy                (N × 384) numpy array
│   │   └── chunk_ids.json             ordered list mapping row index → chunk_id
│   │
│   └── vectordb/                      Phase 4 — ChromaDB on-disk persistence
│       └── chroma/                    auto-managed by chromadb.PersistentClient
│
│
├── src/                               ── all application code (one module per phase)
│   ├── __init__.py
│   │
│   ├── data_loading/                  Phase 1
│   │   ├── __init__.py
│   │   ├── fetcher.py                 HTTP fetch + retry logic
│   │   └── parser.py                  HTML → clean Markdown extraction
│   │
│   ├── chunking/                      Phase 2
│   │   ├── __init__.py
│   │   ├── splitter.py                section-aware + token-boundary splitting
│   │   └── schemas.py                 Chunk dataclass definition
│   │
│   ├── embedding/                     Phase 3
│   │   ├── __init__.py
│   │   └── encoder.py                 MiniLM load + encode + L2-normalize
│   │
│   ├── vector_store/                  Phase 4
│   │   ├── __init__.py
│   │   └── store.py                   ChromaDB collection CRUD + query
│   │
│   ├── retrieval/                     Phase 5
│   │   ├── __init__.py
│   │   ├── guards.py                  PII / advice / returns / corpus guards
│   │   ├── retriever.py               embed query → ChromaDB query → rank
│   │   ├── generator.py               Mistral API call + extractive fallback
│   │   └── pipeline.py                orchestrates guards → retrieve → generate → format
│   │
│   └── ui/                            Streamlit frontend
│       ├── __init__.py
│       ├── app.py                     main entry point: st.chat_message flow
│       └── components.py              welcome banner, examples, disclaimer
│
│
├── tests/                             ── all tests (one file per test category)
│   ├── __init__.py
│   ├── test_retrieval_accuracy.py     Phase 6a — named scheme, ELSS, Gold, ambiguity
│   ├── test_guards.py                 Phase 6b — PII, advice, returns, out-of-corpus, other AMC
│   ├── test_output_format.py          Phase 6c — citation, length, last-updated, no hallucination
│   └── test_edge_cases.py             Phase 6d — empty query, long query, non-English, staleness, multi-scheme
│
│
├── requirements.txt
├── .env.example                       MISTRAL_API_KEY placeholder
└── README.md
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `data/` is git-ignored | Snapshots, chunks, embeddings, and ChromaDB are reproducible artifacts — never commit them. |
| `data/raw/` vs `data/chunks/` | Clear lineage: raw HTML → extracted Markdown (still raw) → chunks. Each phase reads from the previous directory. |
| `data/embeddings/` separate from `data/vectordb/` | Embeddings (`vectors.npy`) are model-version-specific. ChromaDB is the runtime index. Keeping them apart lets you rebuild the index without re-embedding. |
| One module per phase in `src/` | Each phase has a single responsibility. Easy to test in isolation, easy to swap implementations. |
| `pipeline.py` as the orchestrator | Single entry point for the retrieval chain. `app.py` calls `pipeline.run(question)`. Keeps UI thin. |
| `tests/` mirrors phase names | Every test file maps to a specific phase → easy to run targeted: `pytest tests/test_guards.py -v`. |

---

## Technology Stack (PRD-bound)

| Layer | Choice | PRD Ref |
|-------|--------|---------|
| Language | Python 3.11+ | — |
| UI | Streamlit | §6 F9 |
| HTTP | httpx | §4 |
| HTML extraction | trafilatura / BeautifulSoup | §8 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | §7 |
| Vector store | ChromaDB | §7 |
| LLM | Mistral API | §7 |
| Testing | pytest | §11 |

---

*This architecture is scoped to the PRD only. No additional features, datasets, or integrations are planned for v1.*
