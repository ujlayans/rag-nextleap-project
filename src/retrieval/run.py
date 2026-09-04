"""CLI harness for Phase 5 backend retrieval.

Lets you test the full guards → retrieve → generate chain from the terminal
without the UI.

Usage
-----
    python -m src.retrieval.run "What is the expense ratio of SBI Small Cap?"
    python -m src.retrieval.run --show-chunks "ELSS lock-in?"
    # Drill into just guards:
    python -m src.retrieval.run --guards-only "Should I buy gold?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.guards import run_guards
from src.retrieval.pipeline import RetrievalPipeline


def run_query(question: str, show_chunks: bool = False) -> int:
    pipe = RetrievalPipeline()
    # Guards first (both via pipeline and standalone for clarity)
    guard = run_guards(question)
    print("─" * 60)
    print(f"QUESTION : {question}")
    print(f"GUARD    : blocked={guard.blocked} kind={guard.kind or '-'} "
          f"scheme={guard.scheme_name or '-'}")

    answer = pipe.run(question)
    print("─" * 60)
    print(f"ANSWER   : {answer.text}")
    print(f"CITATION : {answer.citation}")
    print(f"UPDATED  : {answer.last_updated}")
    print(f"REFUSED  : {answer.refused} ({answer.refusal_kind or '-'})")
    print(f"SOURCE   : {answer.source}")
    if answer.retrieved:
        print(f"RETRIEVED: {len(answer.retrieved)} chunks")

    if show_chunks and answer.retrieved:
        print("─" * 60)
        print("Top chunks used:")
        for cid in answer.retrieved:
            print(f"  • {cid}")
    print("─" * 60)
    return 0


def guards_only(question: str) -> int:
    guard = run_guards(question)
    print(json.dumps(
        {
            "blocked": guard.blocked,
            "kind": guard.kind,
            "message": guard.message,
            "scheme_name": guard.scheme_name,
            "scheme_url": guard.scheme_url,
            "citation": guard.citation,
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Phase 5 backend retrieval.")
    parser.add_argument("question", nargs="*", help="The question to ask.")
    parser.add_argument("--show-chunks", action="store_true", help="List retrieved chunk ids.")
    parser.add_argument("--guards-only", action="store_true", help="Only run guard pipeline.")
    args = parser.parse_args()

    if args.guards_only:
        sys.exit(guards_only(" ".join(args.question)))
    if not args.question:
        parser.error("Provide a question (e.g. \"What is the expense ratio of SBI Small Cap?\")")
    sys.exit(run_query(" ".join(args.question), show_chunks=args.show_chunks))


if __name__ == "__main__":
    main()
