"""Command-line chat interface for the support agent.

Usage:
    python cli.py

Type your questions; type 'exit' or 'quit' to leave.
"""

from __future__ import annotations

import os

from app.chat_engine import ChatEngine
from app.config import get_settings
from app.llm import get_provider
from app.vectorstore import Retriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    settings = get_settings()
    docs_dir = os.path.join(BASE_DIR, settings.docs_dir)

    retriever = Retriever()
    retriever.build(docs_dir, settings.chunk_size, settings.chunk_overlap)

    if retriever.num_chunks == 0:
        print(f"No documents found in '{docs_dir}'. Add .md/.txt files and retry.")
        return

    engine = ChatEngine(settings, retriever, get_provider(settings))

    print("=" * 60)
    print(" Support Assistant (CLI)")
    print(f" Provider: {settings.llm_provider} | Docs: {retriever.num_docs} | "
          f"Chunks: {retriever.num_chunks}")
    print(" I answer only from the support docs. Type 'exit' to quit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            continue

        result = engine.ask(question)
        print(f"\nAssistant: {result.answer}")
        if result.sources:
            print("\nSources:")
            for s in result.sources:
                print(f"  • {s.doc} — {s.title} (score {s.score})")


if __name__ == "__main__":
    main()
