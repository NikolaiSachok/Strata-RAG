"""Chunking — split documents into small overlapping pieces.

WHY chunk at all? An embedding compresses a whole text into a single fixed-length
vector. If you embed an entire multi-page document, that one vector is a blurry average
of every topic in it — retrieval can't distinguish "the theme section" from "the
features section". Splitting into small chunks gives each passage its own precise
vector, so retrieval can pull *exactly* the relevant paragraph.

WHY overlap? If we cut on a hard boundary, an idea that straddles two chunks loses its
context in both. Overlapping the tail of one chunk with the head of the next means a
boundary-straddling idea still appears intact in at least one chunk. The cost is a
little duplication, which is cheap and worth it.

This is a simple character-window chunker. Production systems often chunk on token
counts or semantic boundaries; the principle is identical and this is easiest to read
and to unit-test (it's pure: text in, chunks out).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit, carrying enough provenance to cite it and to aggregate.

    The metadata travels with the chunk all the way into the vector store's PAYLOAD, so
    retrieval can return *where* a fact came from and aggregation queries can GROUP BY
    project/source-set without re-reading files.
    """
    text: str
    project_id: str
    source_set: str
    source: str        # filename, e.g. "description.docx"
    doc_type: str
    chunk_index: int   # 0-based position within the source document

    @property
    def id(self) -> str:
        """A stable, unique id used to upsert (insert-or-replace) on re-ingest."""
        return f"{self.source_set}/{self.project_id}/{self.source}::{self.chunk_index}"


def chunk_text(text: str, *, size: int, overlap: int) -> list[str]:
    """Split `text` into overlapping character windows. Pure and dependency-free."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    cleaned = text.strip()
    if not cleaned:
        return []
    pieces: list[str] = []
    start = 0
    step = size - overlap  # how far the window advances each step
    while start < len(cleaned):
        piece = cleaned[start : start + size].strip()
        if piece:
            pieces.append(piece)
        start += step
    return pieces
