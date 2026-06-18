"""Tests for the missing-collection PREFLIGHT guard in retrieve.py.

The collection name is corpus-scoped (default = sample → `…_sample`). If a user ingested a REAL
corpus but starts the retriever/API WITHOUT `RAGENGINE_CORPUS_ROOT`, the name falls back to the
non-existent `…_sample` collection. Without the guard, the first `scroll_all` read dies with a
cryptic deep-stack Qdrant 404. These tests prove the guard turns that into a clear, ACTIONABLE
error — and they need NO live Qdrant (the client is a stub whose `collection_exists` returns False).
"""

from __future__ import annotations

import dataclasses

import pytest

from rageval.config import Settings
from rageval.retrieve import CollectionMissingError, Retriever


class _StubClient:
    """A fake Qdrant client: the collection is always absent, and any read would blow up — so the
    test proves the PREFLIGHT runs BEFORE (and instead of) the read."""

    def collection_exists(self, name: str) -> bool:
        return False

    def scroll(self, *a, **k):  # pragma: no cover - must never be reached past the preflight
        raise AssertionError("scroll_all read happened despite a missing collection")


def _settings() -> Settings:
    # Real-ish corpus root + no override so the name is the bare model-derived one; the exact name
    # doesn't matter to the guard (collection_exists is stubbed False either way).
    return dataclasses.replace(Settings.load(), collection_override="")


def test_missing_collection_raises_actionable_error(monkeypatch):
    monkeypatch.setattr("rageval.index.get_client", lambda settings: _StubClient())
    monkeypatch.setattr("rageval.retrieve.get_embedder", lambda settings: object())

    settings = _settings()
    with pytest.raises(CollectionMissingError) as ei:
        Retriever(settings)

    msg = str(ei.value)
    # Names the resolved collection…
    assert settings.collection_name in msg
    # …flags the SAMPLE-corpus default…
    assert "SAMPLE" in msg
    # …and gives BOTH env-var fixes.
    assert "RAGENGINE_CORPUS_ROOT" in msg
    assert "RAGEVAL_COLLECTION" in msg
    # …and how to ingest.
    assert "rageval.ingest" in msg


def test_missing_collection_error_is_a_runtimeerror(monkeypatch):
    # A RuntimeError subclass, so existing `except RuntimeError` paths still catch it — but the
    # error is NOT a raw Qdrant exception (the whole point of the guard).
    monkeypatch.setattr("rageval.index.get_client", lambda settings: _StubClient())
    monkeypatch.setattr("rageval.retrieve.get_embedder", lambda settings: object())

    with pytest.raises(RuntimeError):
        Retriever(_settings())
