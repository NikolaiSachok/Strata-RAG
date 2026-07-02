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

import re
from dataclasses import dataclass

# A page-provenance marker the PDF extractor inserts at each page boundary: `[page N]`.
# Chunking splits it away from a page's later chunks, so we track marker OFFSETS and resolve each
# chunk's page from the last marker at-or-before its start (MAJOR-1). Kept in sync with
# extract/pdf.py's PdfExtraction.text().
_PAGE_MARKER_RE = re.compile(r"\[page (\d+)\]")


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
    page: int | None = None  # 1-based PDF page this chunk's content STARTS on (None for non-PDF)

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


def _page_offsets(text: str) -> list[tuple[int, int]]:
    """Return sorted (char_offset, page_number) for every `[page N]` marker in `text`."""
    return [(m.start(), int(m.group(1))) for m in _PAGE_MARKER_RE.finditer(text)]


def _page_at(offset: int, markers: list[tuple[int, int]]) -> int | None:
    """The page a window STARTING at `offset` belongs to: the last marker at-or-before it.
    None when no marker precedes the offset (non-PDF text, or content before the first marker)."""
    page = None
    for pos, num in markers:
        if pos <= offset:
            page = num
        else:
            break
    return page


def chunk_text_with_pages(text: str, *, size: int, overlap: int) -> list[tuple[str, int | None]]:
    """Like `chunk_text`, but each chunk is paired with the PDF PAGE its content starts on (MAJOR-1).

    The char-window chunker splits a page's `[page N]` marker away from that page's later chunks, so
    the marker alone can't attribute every chunk. We resolve each window's page from the marker
    OFFSETS (the last marker at-or-before the window start) and, when the window doesn't itself begin
    with a marker, PREPEND the resolved `[page N]` so the provenance is visible in the chunk text AND
    in the returned page field. A boundary-spanning window is attributed to the page it STARTS on
    (the correct default — its head content is that page's), never mis-attributed to the next page.

    Returns [] for empty text. Operates on the SAME stripped text `chunk_text` uses, so offsets line
    up with the produced windows.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    cleaned = text.strip()
    if not cleaned:
        return []
    markers = _page_offsets(cleaned)
    out: list[tuple[str, int | None]] = []
    start = 0
    step = size - overlap
    while start < len(cleaned):
        window = cleaned[start : start + size]
        page = _page_at(start, markers)
        piece = window.strip()
        if piece:
            # If this window doesn't already open with a marker for its page, prepend it so the
            # citation is legible in the chunk itself (retrieval shows chunk text). Only when we
            # actually resolved a page (PDF content).
            if page is not None and not _PAGE_MARKER_RE.match(piece):
                piece = f"[page {page}]\n{piece}"
            out.append((piece, page))
        start += step
    return out
