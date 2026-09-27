"""Document ingestion and retrieval (RAG).

Loads support docs from a directory, splits them into overlapping chunks,
and builds a TF-IDF vector index for cosine-similarity retrieval. This works
with no external API keys. The interface (`Retriever`) is intentionally simple
so it can be swapped for an embedding-based vector DB later.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED_EXTS = (".md", ".txt", ".markdown")


@dataclass
class Chunk:
    doc: str          # source file name
    title: str        # nearest heading / doc title
    text: str         # chunk content


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _nearest_heading(text: str, up_to: int) -> str:
    """Find the most recent markdown heading before position `up_to`."""
    headings = list(re.finditer(r"^#{1,6}\s+(.+)$", text[:up_to], flags=re.MULTILINE))
    if headings:
        return headings[-1].group(1).strip()
    return ""


def _split_into_chunks(
    text: str, doc_name: str, chunk_size: int, overlap: int
) -> List[Chunk]:
    chunks: List[Chunk] = []
    text = text.strip()
    if not text:
        return chunks

    # Prefer splitting on blank lines to keep paragraphs intact.
    start = 0
    length = len(text)
    step = max(1, chunk_size - overlap)
    while start < length:
        end = min(start + chunk_size, length)
        # Try to end at a paragraph/sentence boundary near `end`.
        window = text[start:end]
        boundary = max(window.rfind("\n\n"), window.rfind(". "))
        if boundary > chunk_size * 0.5 and end < length:
            end = start + boundary + 1
        piece = text[start:end].strip()
        if piece:
            title = _nearest_heading(text, end) or doc_name
            chunks.append(Chunk(doc=doc_name, title=title, text=piece))
        start += step
    return chunks


class Retriever:
    """TF-IDF backed retriever over the support documentation."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def num_docs(self) -> int:
        return len({c.doc for c in self._chunks})

    def build(self, docs_dir: str, chunk_size: int, overlap: int) -> None:
        paths: List[str] = []
        for ext in SUPPORTED_EXTS:
            paths.extend(glob.glob(os.path.join(docs_dir, "**", f"*{ext}"), recursive=True))
        paths = sorted(set(paths))

        chunks: List[Chunk] = []
        for path in paths:
            rel = os.path.relpath(path, docs_dir)
            content = _read_file(path)
            chunks.extend(_split_into_chunks(content, rel, chunk_size, overlap))

        self._chunks = chunks
        if not chunks:
            self._vectorizer = None
            self._matrix = None
            return

        corpus = [c.text for c in chunks]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if not self._chunks or self._vectorizer is None or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results: List[RetrievedChunk] = []
        for idx in top_idx:
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(scores[idx])))
        return results
