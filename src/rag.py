"""Retrieve chunks and generate a facts-only answer."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    CORPUS_PATH,
    DISCLAIMER,
    EMBEDDING_MODEL,
    MISTRAL_MODEL,
    TOP_K,
)
from src.guards import GuardResult, run_guards

SYSTEM_PROMPT = """You are a facts-only FAQ assistant for seven SBI Mutual Fund Direct Growth schemes listed on Groww.

Rules:
- Answer ONLY using the retrieved context. If the context does not contain the fact, say you do not have it in this corpus.
- Maximum 3 sentences.
- No investment advice, no buy/sell, no portfolio construction.
- Do not compute, annualise, rank, or compare returns. Do not invent NAV, AUM, or ratios.
- Do not ask for or repeat PAN, Aadhaar, account numbers, OTPs, email, or phone.
- Name the scheme clearly.
- Do not add a citation URL in the body (the app appends one).
- Do not add "Last updated" in the body (the app appends it).
"""


def snapshot_date() -> str:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return data.get("snapshot_date", "unknown")


def _collection():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    col = _collection()
    result = col.query(query_texts=[question], n_results=k, include=["documents", "metadatas", "distances"])
    hits = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": doc,
                "url": meta.get("url", ""),
                "scheme": meta.get("scheme", ""),
                "section": meta.get("section", ""),
                "distance": dist,
            }
        )
    return hits


def _mistral_complete(prompt: str) -> str | None:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from mistralai import Mistral

        client = Mistral(api_key=api_key)
        resp = client.chat.complete(
            model=os.getenv("MISTRAL_MODEL", MISTRAL_MODEL),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=220,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"The language model could not be reached ({exc}). Using retrieved facts only."


def _extractive_answer(hits: list[dict], question: str) -> str:
    if not hits:
        return "This corpus does not contain an answer. Ask about one of the seven SBI Direct Growth schemes on Groww."
    top = hits[0]["text"]
    snippet = top.replace("\n", " ")
    if len(snippet) > 420:
        snippet = snippet[:417] + "..."
    return f"From the Groww snapshot for {hits[0]['scheme']}: {snippet}"


def format_answer(body: str, citation: str, updated: str) -> str:
    body = body.strip()
    return f"{body}\n\nSource: {citation}\nLast updated from sources: {updated}"


def answer(question: str) -> dict:
    """Return dict with keys: text, citation, kind, hits, blocked."""
    updated = snapshot_date()
    guard = run_guards(question)
    if guard.blocked:
        text = format_answer(guard.message or "", guard.citation or "", updated)
        return {
            "text": text,
            "citation": guard.citation,
            "kind": guard.kind,
            "hits": [],
            "blocked": True,
            "disclaimer": DISCLAIMER,
        }

    hits = retrieve(question)
    if not hits:
        msg = "No matching scheme text was retrieved. Ask about expense ratio, SIP, exit load, lock-in, or riskometer for a named SBI scheme in this prototype."
        citation = "https://groww.in/mutual-funds/sbi-small-midcap-fund-direct-growth"
        return {
            "text": format_answer(msg, citation, updated),
            "citation": citation,
            "kind": "no_hits",
            "hits": [],
            "blocked": False,
            "disclaimer": DISCLAIMER,
        }

    context = "\n\n---\n\n".join(
        f"[{h['scheme']} | {h['section']} | {h['url']}]\n{h['text']}" for h in hits
    )
    user_prompt = f"Question: {question}\n\nRetrieved context:\n{context}"
    generated = _mistral_complete(user_prompt)
    if not generated:
        generated = _extractive_answer(hits, question)

    citation = hits[0]["url"]
    return {
        "text": format_answer(generated, citation, updated),
        "citation": citation,
        "kind": "rag",
        "hits": hits,
        "blocked": False,
        "disclaimer": DISCLAIMER,
    }


# Re-export for tests
GuardResult = GuardResult
