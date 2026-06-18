"""Deterministic tests for the chunker.

Chunking is pure (text in, chunks out) with no model, no I/O — so its behaviour is fully
predictable and a perfect thing to lock down. These tests document the chunker's
contract: size, overlap, and edge cases.
"""

from __future__ import annotations

import pytest

from rageval.chunking import Chunk, chunk_text


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
