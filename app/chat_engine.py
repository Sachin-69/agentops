"""Chat engine that ties retrieval + LLM together with strict guardrails.

Guardrails enforced here (the core requirement):
  1. Retrieve relevant chunks from the support docs.
  2. If NOTHING clears the relevance threshold -> refuse (out of scope).
  3. Only the retrieved doc context is passed to the LLM.
  4. If the LLM signals it cannot answer from context -> refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import Settings
from .llm import OUT_OF_SCOPE_TOKEN, LLMProvider
from .vectorstore import Retriever, RetrievedChunk


@dataclass
class Source:
    doc: str
    title: str
    score: float
    snippet: str


@dataclass
class ChatResult:
    answer: str
    grounded: bool          # True if answer came from the docs
    sources: List[Source]


class ChatEngine:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        provider: LLMProvider,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._provider = provider

    def _build_context(self, hits: List[RetrievedChunk]) -> str:
        blocks: List[str] = []
        for i, hit in enumerate(hits, start=1):
            header = f"[Source {i}] {hit.chunk.doc} — {hit.chunk.title}"
            blocks.append(f"{header}\n{hit.chunk.text}")
        return "\n\n---\n\n".join(blocks)

    def _sources(self, hits: List[RetrievedChunk]) -> List[Source]:
        sources: List[Source] = []
        for hit in hits:
            snippet = hit.chunk.text.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220].rsplit(" ", 1)[0] + "…"
            sources.append(
                Source(
                    doc=hit.chunk.doc,
                    title=hit.chunk.title,
                    score=round(hit.score, 4),
                    snippet=snippet,
                )
            )
        return sources

    def ask(self, question: str) -> ChatResult:
        question = (question or "").strip()
        if not question:
            return ChatResult(
                answer="Please enter a question.",
                grounded=False,
                sources=[],
            )

        hits = self._retriever.search(question, self._settings.top_k)
        relevant = [h for h in hits if h.score >= self._settings.min_relevance]

        # Guardrail 1: nothing relevant in the docs -> refuse.
        if not relevant:
            return ChatResult(
                answer=self._settings.refusal_message,
                grounded=False,
                sources=[],
            )

        context = self._build_context(relevant)
        raw_answer = self._provider.generate(question, context).strip()

        # Guardrail 2: LLM says it can't answer from the context -> refuse.
        if not raw_answer or OUT_OF_SCOPE_TOKEN in raw_answer:
            return ChatResult(
                answer=self._settings.refusal_message,
                grounded=False,
                sources=[],
            )

        return ChatResult(
            answer=raw_answer,
            grounded=True,
            sources=self._sources(relevant),
        )
