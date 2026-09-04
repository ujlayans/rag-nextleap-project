"""Generator: produce a facts-only answer from retrieved chunks.

Primary path: Mistral API (if ``MISTRAL_API_KEY`` is set in env / .env).
Fallback chain: Groq API (if ``GROQ_API_KEY`` is set), then extractive —
return the most relevant chunk verbatim, prefixed with its source URL
(PRD §7, architecture §5 "Fallback (no Mistral key)").
"""

from __future__ import annotations

import logging
import os

from dataclasses import dataclass

from dotenv import load_dotenv

from src.config import GROQ_MODEL, MISTRAL_MODEL

logger = logging.getLogger(__name__)


@dataclass
class GenResult:
    """A generated answer plus the provider that produced it."""

    text: str
    provider: str  # "mistral" | "groq" | "extractive"

PROMPT_TEMPLATE = """You are a facts-only assistant for SBI Mutual Fund schemes on Groww.

RULES:
1. Answer in <=3 sentences using ONLY the provided context.
2. Do not include source brackets like "[Source: ...]" or markdown links — just the plain answer text.
3. End with: "Last updated from sources: <date>" using a real date from the context below.
4. If the context does not contain the answer, say: "I don't have this information in my sources."
5. Never give buy/sell advice. Never compare returns. Never invent numbers.

Context:
{chunks_with_urls}

Question: {user_question}
"""


def _chunk_block(entry) -> str:
    return (
        f"[Source: {entry.source_url} | Section: {entry.section} | "
        f"Snapshot date: {entry.snapshot_date}]\n{entry.text}\n"
    )


def build_prompt(question: str, chunks) -> str:
    context = "\n\n".join(_chunk_block(c) for c in chunks)
    return PROMPT_TEMPLATE.format(chunks_with_urls=context, user_question=question)


def generator_available() -> bool:
    load_dotenv()
    return bool(os.getenv("MISTRAL_API_KEY") or os.getenv("GROQ_API_KEY"))


def generate(
    question: str,
    chunks,
    *,
    model: str = MISTRAL_MODEL,
) -> GenResult:
    """Return a plain-text answer plus the provider used.

    Provider chain (first success wins):
      1. Mistral   — if ``MISTRAL_API_KEY`` is set.
      2. Groq      — if ``GROQ_API_KEY`` is set (runs when Mistral is absent
                     or errors, e.g. the per-workspace 429 rate limit).
      3. Extractive — always available fallback (architecture §5).
    """
    load_dotenv()
    mistral_key = os.getenv("MISTRAL_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if mistral_key:
        try:
            text = _generate_mistral(mistral_key, question, chunks, model).strip()
            if text:
                return GenResult(text=text, provider="mistral")
        except Exception as exc:  # degrade to the next provider
            logger.warning("Mistral generation failed (%s); falling back.", exc)

    if groq_key:
        try:
            text = _generate_groq(groq_key, question, chunks).strip()
            if text:
                return GenResult(text=text, provider="groq")
        except Exception as exc:
            logger.warning("Groq generation failed (%s); using extractive fallback.", exc)

    return GenResult(text=_generate_extractive(chunks), provider="extractive")


def _generate_extractive(chunks) -> str:
    """Fallback: return the single most relevant chunk verbatim + citation.

    Picks the cosine top-1 retrieved chunk (architecture §5 "Fallback").
    """
    if not chunks:
        return (
            "I don't have this information in my sources. "
            "Ask a factual question about one of the seven SBI schemes."
        )
    best = chunks[0]
    return (
        f"{best.text.strip()}\n\n"
        f"Source: {best.source_url}\n"
        f"Last updated from sources: {best.snapshot_date}"
    )


def _generate_mistral(api_key: str, question: str, chunks, model: str) -> str:
    from mistralai import Mistral

    prompt = build_prompt(question, chunks)
    client = Mistral(api_key=api_key)
    resp = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": "You answer factual questions. Follow the rules."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def _generate_groq(api_key: str, question: str, chunks) -> str:
    from groq import Groq

    prompt = build_prompt(question, chunks)
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You answer factual questions. Follow the rules."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=700,
    )
    text = resp.choices[0].message.content.strip()
    # Defensive: strip any leftover reasoning block (e.g. "thinking").
    if "\n" in text and text.lstrip().lower().startswith("thinking"):
        text = text.split("\n", 1)[1].strip()
    return text
