"""Tests for the roster loader + the app_name/publisher split + --enrich-only.

The DECISION under test: the project's PRODUCT name (`app_name`, LLM/config-extracted) is a
DIFFERENT facet from the authoritative publisher (`publisher`, a deterministic roster TSV join).
publisher is NEVER inferred by the LLM; it comes from a roster TSV keyed on the leading numeric
project id, and is null when no roster row / no roster file (graceful degradation). The publisher
can DIFFER from the product title, and when it does we trust the roster over LLM inference.

All values here are FICTIONAL (data/sample/*.tsv) — no real names/ids ever appear in tests.
"""

from __future__ import annotations

import dataclasses

from pathlib import Path

from rageval.roster import Roster, extract_numeric_id
from rageval.config import SAMPLE_ROSTER_DIR, SETTINGS
from rageval.enrich import enrich_project
from rageval.sources.base import SourceDoc


# --- numeric-id extraction handles the messy id shapes -----------------------

def test_extract_numeric_id_handles_suffixed_and_padded_ids():
    assert extract_numeric_id("2268") == 2268
    assert extract_numeric_id("1490-sp08") == 1490      # leading id, suffix ignored
    assert extract_numeric_id("2288_Summit (extra)") == 2288
    assert extract_numeric_id("0018") == 18             # zero-pad → same int as "18"
    assert extract_numeric_id("18") == 18
    assert extract_numeric_id("atlas-ledger") is None   # no digits → no roster row
    assert extract_numeric_id("") is None


# --- the TSV join populates publisher from fictional id → fictional publisher ---

def _sample_roster() -> Roster:
    return Roster(SAMPLE_ROSTER_DIR)


def test_tsv_join_populates_publisher():
    r = _sample_roster()
    # northwind.tsv: 0001 → 'Maple Lagoon' (fictional).
    assert r.publisher("northwind", "0001") == "Maple Lagoon"
    assert r.publisher("northwind", "0004") == "Velvet Summit"


def test_publisher_null_when_no_matching_row():
    r = _sample_roster()
    # 9999 is not a row in northwind.tsv → null (intentionally incomplete roster, not an error).
    assert r.publisher("northwind", "9999") is None


def test_publisher_null_when_no_roster_file():
    r = _sample_roster()
    # An unmapped source_set family resolves to null (a corpus need not supply a roster:
    # publisher is simply null, identical code path).
    assert r.publisher("custom", "1500") is None
    # An unmapped source_set family also yields null.
    assert r.publisher("unknown-set", "0001") is None


def test_publisher_matches_suffixed_id_via_leading_numeric():
    r = _sample_roster()
    # atlas.tsv has id 1490 → 'Velvet Summit'; a suffixed engine project_id still joins.
    assert r.publisher("atlas", "1490-sp08") == "Velvet Summit"
    assert r.publisher("atlas", "2288_Summit (x)") == "Amber Hollow"


def test_family_mapping_shares_one_tsv_across_subsets():
    """All subsets of a family map to the SAME family file (resolved via the leading stem). The
    'atlas' family shares one atlas.tsv, so every atlas* subset joins id 1490 to the same row."""
    r = _sample_roster()
    for ss in ("atlas", "atlas-extra", "atlas-archive", "atlas-requirements"):
        assert r.publisher(ss, "1490-sp08") == "Velvet Summit"  # one shared atlas.tsv


# --- app_name (config/LLM) is DISTINCT from publisher (TSV) ------------------

def _doc(name: str, doc_type: str, text: str, project_id: str) -> SourceDoc:
    return SourceDoc(project_id=project_id, source_set="northwind", doc_path=Path(name),
                     doc_type=doc_type, ext=name.rsplit(".", 1)[-1], raw_text=text)


def test_app_name_and_publisher_are_distinct_fields():
    """app_name comes from settings.md/LLM (the surface product title); publisher comes from the
    roster TSV join — they are independent and need not coincide. This is the conflation lesson:
    the LLM-inferred title ('Citrus Garden') differs from the roster publisher ('Maple Lagoon'),
    so we trust the roster (ground truth) for the publisher facet."""
    r = _sample_roster()
    docs = [_doc("settings.md", "metadata",
                 "Brand: Citrus Garden\nCategory: to-do\nTheme: citrus", project_id="0001")]
    rec = enrich_project(None, "northwind", "0001", docs, chunk_count=1, roster=r)
    # The doc-stated product title → app_name.
    assert rec.app_name == "Citrus Garden"
    # The publisher is the authoritative TSV value — DIFFERENT from app_name.
    assert rec.publisher == "Maple Lagoon"
    assert rec.app_name != rec.publisher


def test_publisher_null_for_named_project_without_numeric_id():
    r = _sample_roster()
    docs = [_doc("README.md", "description", "A simple named project.", project_id="atlas-ledger")]
    rec = enrich_project(None, "atlas", "atlas-ledger", docs, chunk_count=1, roster=r)
    assert rec.publisher is None   # no numeric id → no TSV row


# --- --enrich-only: rewrites the sidecar WITHOUT touching Qdrant -------------

def test_enrich_only_rewrites_sidecar_without_touching_qdrant(tmp_path, monkeypatch):
    """--enrich-only re-runs enrichment + harvest + roster join and rewrites the sidecar, but
    NEVER embeds/upserts. We prove the index is untouched by making any index call EXPLODE — the
    run must still complete and populate the sidecar (incl. publisher on a mapped project)."""
    import rageval.ingest as ingest_mod
    from rageval.sidecar import all_projects, connect

    # Any attempt to use the Qdrant index would import these — make them blow up if called.
    def _boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("enrich-only must NOT touch the Qdrant index")

    import rageval.index as index_mod
    monkeypatch.setattr(index_mod, "get_client", _boom, raising=False)
    monkeypatch.setattr(index_mod, "ensure_collection", _boom, raising=False)
    monkeypatch.setattr(index_mod, "upsert_chunks", _boom, raising=False)

    # Isolate the sidecar to a temp DB (enrich_only's connect() with no arg uses SIDECAR_PATH).
    db = tmp_path / "side.sqlite"
    monkeypatch.setattr(ingest_mod, "connect", lambda *a, **k: connect(db))

    # Run over the fictional sample corpus, structural-only (no LLM) → deterministic.
    settings = dataclasses.replace(SETTINGS, enrich_concurrency=4)
    summary = ingest_mod.enrich_only(settings, use_enrich=False)
    assert summary["projects_enriched"] > 0

    recs = {r.key: r for r in all_projects(connect(db))}
    # The sample northwind 0001 is in northwind.tsv → publisher populated from the join.
    assert "northwind/0001" in recs
    assert recs["northwind/0001"].publisher == "Maple Lagoon"
