"""Turn the closed JSON snapshot into text chunks and persist in ChromaDB."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHROMA_DIR, COLLECTION_NAME, CORPUS_PATH, EMBEDDING_MODEL


def load_corpus() -> dict:
    return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))


def build_chunks(corpus: dict) -> list[dict]:
    chunks: list[dict] = []
    snapshot = corpus["snapshot_date"]
    nav_as_of = corpus["nav_as_of"]

    for scheme in corpus["schemes"]:
        facts = (
            f"Scheme: {scheme['name']}\n"
            f"AMC: SBI Mutual Fund. Plan: Direct Growth.\n"
            f"Category: {scheme['category']}. Asset class: {scheme['asset_class']}.\n"
            f"Riskometer: {scheme['riskometer']}.\n"
            f"Expense ratio: {scheme['expense_ratio']}.\n"
            f"NAV: {scheme['nav']} as of {nav_as_of}.\n"
            f"Fund size (AUM): {scheme['aum']}.\n"
            f"Minimum SIP: {scheme['min_sip']}.\n"
            f"Minimum for 1st lumpsum investment: {scheme['min_lumpsum_first']}.\n"
            f"Minimum for additional lumpsum: {scheme['min_lumpsum_additional']}.\n"
            f"Exit load: {scheme['exit_load']}.\n"
            f"Lock-in: {scheme['lock_in']}.\n"
            f"Benchmark: {scheme['benchmark']}.\n"
            f"Groww rating (stars): {scheme['rating_groww']}.\n"
            f"Source URL: {scheme['url']}\n"
            f"Snapshot date: {snapshot}."
        )
        chunks.append(
            {
                "id": f"{scheme['id']}_facts",
                "text": facts,
                "url": scheme["url"],
                "scheme": scheme["name"],
                "section": "key_facts",
            }
        )

        about = (
            f"About {scheme['name']}: {scheme['investment_objective']}\n"
            f"Fund managers: {scheme['fund_managers']}.\n"
            f"SBI Mutual Fund website: {scheme['amc_website']}. Phone (as on Groww): {scheme['amc_phone']}.\n"
            f"Source URL: {scheme['url']}. Snapshot date: {snapshot}."
        )
        if scheme.get("holdings_note"):
            about += f"\nHoldings note: {scheme['holdings_note']}"
        chunks.append(
            {
                "id": f"{scheme['id']}_about",
                "text": about,
                "url": scheme["url"],
                "scheme": scheme["name"],
                "section": "about",
            }
        )

        costs = (
            f"Costs and tax notes for {scheme['name']}.\n"
            f"Expense ratio: {scheme['expense_ratio']}.\n"
            f"Exit load: {scheme['exit_load']}.\n"
            f"Stamp duty on investment: {scheme['stamp_duty']}.\n"
            f"Tax implication as printed on Groww: {scheme['tax_implication']}\n"
            f"Lock-in: {scheme['lock_in']}.\n"
            f"Source URL: {scheme['url']}. Snapshot date: {snapshot}."
        )
        chunks.append(
            {
                "id": f"{scheme['id']}_costs",
                "text": costs,
                "url": scheme["url"],
                "scheme": scheme["name"],
                "section": "costs_tax",
            }
        )

    for i, item in enumerate(corpus.get("glossary", [])):
        chunks.append(
            {
                "id": f"glossary_{i}",
                "text": f"Definition — {item['term']}: {item['text']}\nSource URL: {item['url']}. Snapshot date: {snapshot}.",
                "url": item["url"],
                "scheme": "glossary",
                "section": "glossary",
            }
        )
    return chunks


def ingest() -> int:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    corpus = load_corpus()
    chunks = build_chunks(corpus)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"url": c["url"], "scheme": c["scheme"], "section": c["section"]} for c in chunks],
    )
    print(f"Ingested {len(chunks)} chunks into {CHROMA_DIR} ({COLLECTION_NAME}).")
    return len(chunks)


if __name__ == "__main__":
    ingest()
