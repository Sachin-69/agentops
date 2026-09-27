"""Smoke tests for the docs-grounded agent.

Run with:  python test_agent.py
Verifies the core requirement: in-scope questions get answered from docs,
out-of-scope questions are refused.
"""

from __future__ import annotations

import os
import sys

from app.chat_engine import ChatEngine
from app.config import get_settings
from app.llm import get_provider
from app.vectorstore import Retriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_engine() -> ChatEngine:
    settings = get_settings()
    docs_dir = os.path.join(BASE_DIR, settings.docs_dir)
    retriever = Retriever()
    retriever.build(docs_dir, settings.chunk_size, settings.chunk_overlap)
    assert retriever.num_chunks > 0, "No docs indexed!"
    return ChatEngine(settings, retriever, get_provider(settings))


def main() -> int:
    engine = build_engine()
    failures = 0

    in_scope = [
        "How do I get a refund?",
        "What payment methods do you accept?",
        "I forgot my password, what do I do?",
        "How much does the Pro plan cost?",
    ]
    out_of_scope = [
        "What is the capital of France?",
        "Can you write me a poem about the ocean?",
        "What's the weather tomorrow?",
        "Who won the world cup in 2018?",
    ]

    print("== IN-SCOPE (should be grounded) ==")
    for q in in_scope:
        r = engine.ask(q)
        ok = r.grounded
        print(f"[{'PASS' if ok else 'FAIL'}] {q}")
        if not ok:
            failures += 1

    print("\n== OUT-OF-SCOPE (should be refused) ==")
    for q in out_of_scope:
        r = engine.ask(q)
        ok = not r.grounded
        print(f"[{'PASS' if ok else 'FAIL'}] {q}")
        if not ok:
            failures += 1

    print(f"\nResult: {'ALL PASSED' if failures == 0 else str(failures) + ' FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
