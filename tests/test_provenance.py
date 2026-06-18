"""Provenance / lineage tests — folder-derived metadata must reach the sidecar.

An adapter can surface structural PROVENANCE in `folder_meta`: a port/predecessor lineage link,
and a non-conforming status/flag for marker projects. These come from folder structure (not the
LLM), so they must populate the sidecar EVEN when enrichment is skipped (use_llm=False). These
tests lock that flow down, plus the SQLite round-trip of the provenance columns. Deterministic,
no model, no network — they use a temp sqlite file. All ids/names are fictional.
"""

from __future__ import annotations

from pathlib import Path

from rageval.enrich import enrich_all
from rageval.sidecar import ProjectRecord, all_projects, connect, upsert_project
from rageval.sources.base import SourceDoc


def _doc(source_set, project_id, doc_type, meta, *, ext="md", text="x" * 60):
    return SourceDoc(project_id=project_id, source_set=source_set, doc_path=Path(f"{project_id}"),
                     doc_type=doc_type, ext=ext, raw_text=text, folder_meta=meta)


def test_lineage_reaches_record_without_llm():
    docs = [_doc("northwind", "2023", "spec",
                 {"project_dir": "2023", "brand_hint": "Aurora Drift", "kotlin_source_id": "1798"})]
    recs = enrich_all(docs, {("northwind", "2023"): 1}, use_llm=False)
    rec = next(r for r in recs if r.project_id == "2023")
    assert rec.kotlin_source_id == "1798"          # lineage survives with NO enrichment
    assert rec.non_conforming is None and rec.status is None


def test_marker_status_reaches_record_without_llm():
    docs = [_doc("northwind-archive", "1788", "marker",
                 {"project_dir": "1788", "status": "banned", "non_conforming": True, "name": "NeonDrift"},
                 ext="marker", text="Project 1788 — non-conforming. Status: banned.")]
    recs = enrich_all(docs, {("northwind-archive", "1788"): 1}, use_llm=False)
    rec = recs[0]
    assert rec.status == "banned"
    assert rec.non_conforming is True
    assert rec.doc_types == ["marker"]


def test_sidecar_round_trips_provenance_columns(tmp_path):
    db = tmp_path / "side.sqlite"
    conn = connect(db)
    upsert_project(conn, ProjectRecord(
        project_id="2023", source_set="northwind", app_name="Aurora Drift",
        publisher="Neon Spins", kotlin_source_id="1798", chunk_count=1))
    upsert_project(conn, ProjectRecord(
        project_id="1788", source_set="northwind-archive", status="banned",
        non_conforming=True, doc_types=["marker"], chunk_count=1))
    conn.close()

    conn2 = connect(db)
    recs = {r.key: r for r in all_projects(conn2)}
    # Lineage query: "what was project 2023 ported from?"
    assert recs["northwind/2023"].kotlin_source_id == "1798"
    # The two distinct name facets round-trip independently.
    assert recs["northwind/2023"].app_name == "Aurora Drift"
    assert recs["northwind/2023"].publisher == "Neon Spins"
    # Conformance query: "which projects are banned / non-conforming?"
    banned = [r for r in recs.values() if r.status == "banned"]
    assert [r.project_id for r in banned] == ["1788"]
    assert recs["northwind-archive/1788"].non_conforming is True
    conn2.close()
