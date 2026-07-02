"""Deterministic tests for the chunker.

Chunking is pure (text in, chunks out) with no model, no I/O — so its behaviour is fully
predictable and a perfect thing to lock down. These tests document the chunker's
contract: size, overlap, and edge cases.
"""

from __future__ import annotations

import pytest

from rageval.chunking import Chunk, chunk_text, chunk_text_with_pages


def test_short_text_is_one_chunk():
    pieces = chunk_text("hello world", size=800, overlap=150)
    assert pieces == ["hello world"]


def test_chunks_advance_by_size_minus_overlap():
    # 1000 chars, size 400, overlap 100 → step 300 → starts at 0,300,600,900.
    text = "".join(str(i % 10) for i in range(1000))
    pieces = chunk_text(text, size=400, overlap=100)
    assert len(pieces) == 4
    # Adjacent chunks overlap by `overlap` chars: tail of piece0 == head of piece1.
    assert pieces[0][-100:] == pieces[1][:100]


def test_no_chunk_exceeds_size():
    pieces = chunk_text("abc " * 1000, size=300, overlap=50)
    assert all(len(p) <= 300 for p in pieces)


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n  ", size=300, overlap=50) == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("anything", size=100, overlap=100)


def test_chunk_id_is_stable_and_unique():
    a = Chunk(text="t", project_id="0001", source_set="northwind",
              source="overview.md", doc_type="spec", chunk_index=0)
    b = Chunk(text="t", project_id="0001", source_set="northwind",
              source="overview.md", doc_type="spec", chunk_index=1)
    assert a.id == "northwind/0001/overview.md::0"
    assert a.id != b.id


def test_chunk_is_frozen_dataclass():
    c = Chunk(text="t", project_id="0001", source_set="northwind",
              source="s.md", doc_type="spec", chunk_index=0)
    with pytest.raises(Exception):
        c.text = "mutated"  # type: ignore[misc]


# ===========================================================================
# MAJOR-1 — page provenance survives chunking (every chunk resolves to its page).
# ===========================================================================

def test_page_provenance_survives_chunking_on_multi_chunk_page():
    """A `[page N]` marker is inserted once per page, but a page long enough to split into several
    chunks must attribute EVERY chunk to the right page — not just its first. Later chunks of page 1
    stay page 1; page 2's chunks are page 2 (no tail mis-attribution)."""
    page1 = "A" * 300          # page 1 body: long enough to span multiple windows
    page2 = "B" * 300
    text = f"[page 1]\n{page1}\n\n[page 2]\n{page2}"
    pairs = chunk_text_with_pages(text, size=100, overlap=20)
    for piece, page in pairs:
        assert page in (1, 2)
        # A chunk is attributed to the page its content STARTS on: the first body char after the
        # marker head matches that page's letter (a boundary-spanning chunk may also carry the next
        # page's content, but is correctly attributed to where it began — never mis-attributed).
        body = piece.split("]", 1)[1].strip() if piece.startswith("[page") else piece.strip()
        first = body.lstrip("\n")[:1]
        assert first == ("A" if page == 1 else "B")
    # A LATER chunk of page 1 (not just its first) is still page 1 — the core MAJOR-1 guarantee.
    assert sum(1 for _, p in pairs if p == 1) >= 2
    assert sum(1 for _, p in pairs if p == 2) >= 2


def test_every_chunk_carries_a_page_marker_in_text():
    """Each chunk's TEXT carries a legible `[page N]` head (so the retrieved citation shows a page),
    even the ones the raw marker was split away from."""
    text = "[page 5]\n" + ("X" * 250)
    pairs = chunk_text_with_pages(text, size=100, overlap=10)
    assert len(pairs) >= 2
    assert all(piece.startswith("[page 5]") for piece, _ in pairs)
    assert all(page == 5 for _, page in pairs)


def test_non_pdf_text_has_no_page():
    """Text with no page markers (a .md/.docx doc) yields page=None for every chunk."""
    pairs = chunk_text_with_pages("plain markdown content here", size=800, overlap=100)
    assert pairs and all(page is None for _, page in pairs)
