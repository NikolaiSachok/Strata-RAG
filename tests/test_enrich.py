"""Tests for metadata enrichment — settings.md routing + best-available extraction.

These exercise the DECISION: settings.md is METADATA fed to enrich (not embedded), enrichment
PREFERS its structured fields when present, and DEGRADES GRACEFULLY (still produces metadata
from other docs) when no settings.md exists. All deterministic — no LLM (enrich_project with
llm=None uses the structural+structured fallback path)."""

from __future__ import annotations

import dataclasses
import threading

from pathlib import Path

from rageval.config import SETTINGS, SAMPLE_CORPUS_DIR
from rageval.enrich import (
    _best_metadata_source, _parse_metadata_fields, enrich_all, enrich_project,
)
from rageval.sources.base import SourceDoc


def _doc(name: str, doc_type: str, text: str, project_id: str = "p1",
         folder_meta: dict | None = None) -> SourceDoc:
    return SourceDoc(
        project_id=project_id, source_set="atlas", doc_path=Path(name),
        doc_type=doc_type, ext=name.rsplit(".", 1)[-1], raw_text=text,
        folder_meta=folder_meta or {},
    )


def test_parse_metadata_fields_extracts_known_keys():
    text = "Brand: Lemon Ledger\nCategory: budgeting\nTheme: citrus\nMascot: Zest\nrandom: x"
    fields = _parse_metadata_fields(text)
    # A settings.md Brand:/Name: line names the PRODUCT → app_name (not the publisher).
    assert fields["app_name"] == "Lemon Ledger"
    assert fields["app_category"] == "budgeting"
    assert fields["theme"] == "citrus"
    assert fields["mascot"] == "Zest"
    assert "random" not in fields


def test_parse_metadata_ignores_redacted_values():
    # A scrubbed secret line must NOT be lifted as a field value.
    fields = _parse_metadata_fields("api_key: [REDACTED_KEY]\nBrand: Keep Me")
    assert fields.get("app_name") == "Keep Me"
    assert "api_key" not in fields


def test_best_metadata_source_prefers_settings_md():
    docs = [
        _doc("description.md", "description", "A budgeting app about lemons."),
        _doc("settings.md", "metadata", "Brand: Lemon Ledger\nCategory: budgeting\nTheme: citrus"),
    ]
    fields = _best_metadata_source(docs)
    assert fields["app_name"] == "Lemon Ledger"
    assert fields["app_category"] == "budgeting"
    assert fields["theme"] == "citrus"


def test_enrich_fallback_uses_settings_md_structured_fields_without_llm():
    """With NO LLM, enrich still fills app_name/category/theme from settings.md (structured), so
    its fields reach the sidecar even though settings.md is never embedded."""
    docs = [
        _doc("overview.md", "spec", "Some narrative about the app."),
        _doc("settings.md", "metadata", "Brand: Lemon Ledger\nCategory: budgeting\nTheme: citrus"),
    ]
    rec = enrich_project(None, "atlas", "p1", docs, chunk_count=3)
    assert rec.app_name == "Lemon Ledger"
    assert rec.app_category == "budgeting"
    assert "citrus" in rec.theme_tags
    # The metadata doc_type is recorded (sidecar observability), proving it reached enrich.
    assert "metadata" in rec.doc_types


def test_enrich_degrades_gracefully_without_settings_md():
    """A project WITHOUT a settings.md must still produce a record (structural fields), proving
    enrichment does NOT depend on settings.md existing."""
    docs = [_doc("overview.md", "spec", "A clean product description with no metadata file.")]
    rec = enrich_project(None, "atlas", "p2", docs, chunk_count=2)
    assert rec.project_id == "p2"
    assert rec.doc_types == ["spec"]
    assert rec.chunk_count == 2
    # No structured source → app_name/category stay None (honest null), not invented.
    assert rec.app_name is None and rec.app_category is None


def test_enrich_seeds_app_name_from_folder_hint_when_no_settings():
    """An adapter may encode the product name/theme in the FILENAME (folder_meta), so enrich can
    still seed structured fields with no settings.md and no LLM."""
    docs = [_doc("2285 BowlMaster (Bowling).md", "description", "Bowling score manager.",
                 folder_meta={"brand_hint": "BowlMaster", "theme_hint": "Bowling Score Manager"})]
    rec = enrich_project(None, "northwind-spec", "2285", docs, chunk_count=1)
    assert rec.app_name == "BowlMaster"
    assert "bowling score manager" in rec.theme_tags


# --- config.yaml harvest reaches the enriched record (no LLM) ---------------

def _sample_doc(project: str, name: str, doc_type: str = "promo") -> SourceDoc:
    """A SourceDoc whose doc_path lives under the real sample project dir, so enrich can recover
    the project directory and harvest its back/config.yaml."""
    path = SAMPLE_CORPUS_DIR / "atlas" / project / "back" / name
    return SourceDoc(project_id=project, source_set="atlas", doc_path=path,
                     doc_type=doc_type, ext=name.rsplit(".", 1)[-1], raw_text="content")


def test_enrich_harvests_config_yaml_into_record():
    """atlas-vista has a back/config.yaml → domain/landing_url/app_name reach the record (no LLM),
    and the bounded contact-email is derived AND flagged."""
    rec = enrich_project(None, "atlas", "atlas-vista", [_sample_doc("atlas-vista", "index.php")],
                         chunk_count=1)
    assert rec.domain == "vista-weather-7011.test"
    assert rec.landing_url == "https://vista-weather-7011.test"
    assert rec.app_name == "Vista Weather"
    assert rec.app_number == "7011"
    assert rec.contact_emails == ["support@vista-weather-7011.test"]
    assert rec.contact_emails_derived is True
    # config.yaml app.name is the app_name (product/display name).
    assert rec.app_name == "Vista Weather"


def test_enrich_harvest_never_writes_secrets_to_record():
    """The config.yaml fixture carries fake secret blocks — none may appear in the record."""
    rec = enrich_project(None, "atlas", "atlas-vista", [_sample_doc("atlas-vista", "index.php")],
                         chunk_count=1)
    blob = repr(rec).lower()
    assert "fake_value_never_harvested" not in blob
    assert "analytics" not in blob and "integration" not in blob and "session_id" not in blob


def test_enrich_settings_md_app_name_yields_to_config_yaml_when_present():
    """atlas-ledger has BOTH a settings.md (Brand: Lemon Ledger → app_name) and a config.yaml.
    config.yaml app.name is the TOP-priority source for app_name, so the harvested app.name wins;
    the harvest also supplies the authoritative domain/landing_url. (When config.yaml lacks an
    app.name, the settings.md value would stand — see the no-config path test.)"""
    docs = [
        _doc("settings.md", "metadata", "Brand: Lemon Ledger\nCategory: budgeting\nTheme: citrus",
             project_id="atlas-ledger"),
        _sample_doc("atlas-ledger", "config.yaml"),  # provides project dir for the harvest
    ]
    rec = enrich_project(None, "atlas", "atlas-ledger", docs, chunk_count=2)
    # config.yaml app.name ('Lemon Ledger') is the app_name; it is a PRODUCT name, never the
    # publisher. (Here settings.md and config.yaml agree on the product name.)
    assert rec.app_name == "Lemon Ledger"
    assert rec.app_category == "budgeting"       # settings.md category stands
    assert rec.domain == "lemon-ledger-7022.test"
    assert rec.landing_url == "https://lemon-ledger-7022.test"


def test_enrich_no_config_yaml_leaves_harvest_fields_none():
    """A project with no back/config.yaml (atlas-orchard) leaves domain/landing_url None — proving
    graceful degradation, no fabrication."""
    docs = [_doc("description.md", "description", "A simple app.", project_id="atlas-orchard")]
    # doc_path here is a bare Path (no project-dir ancestor) → harvest finds nothing.
    rec = enrich_project(None, "atlas", "atlas-orchard", docs, chunk_count=1)
    assert rec.domain is None and rec.landing_url is None and rec.app_name is None
    assert rec.contact_emails == []


# --- concurrent enrichment (ThreadPool): equivalence, isolation, thread-safety -----------

# A small set of independent projects to enrich in parallel.
def _multi_project_docs() -> list[SourceDoc]:
    return [
        _doc("settings.md", "metadata",
             "Brand: Lemon Ledger\nCategory: budgeting\nTheme: citrus", project_id="p1"),
        _doc("settings.md", "metadata",
             "Brand: Space Timer\nCategory: focus timer\nTheme: space", project_id="p2"),
        _doc("desc.md", "description", "A plain app with no metadata.", project_id="p3"),
        _doc("2285 BowlMaster (Bowling).md", "description", "Bowling score manager.",
             project_id="p4", folder_meta={"brand_hint": "BowlMaster", "theme_hint": "Bowling"}),
    ]


def _chunk_counts(docs: list[SourceDoc]) -> dict[tuple[str, str], int]:
    return {(d.source_set, d.project_id): 1 for d in docs}


def _sequential_records(docs, chunk_counts):
    """The reference: enrich each project one at a time with the same (no-LLM) path."""
    out = []
    seen = set()
    for d in docs:
        key = (d.source_set, d.project_id)
        if key in seen:
            continue
        seen.add(key)
        group = [x for x in docs if (x.source_set, x.project_id) == key]
        out.append(enrich_project(None, d.source_set, d.project_id, group,
                                  chunk_counts.get(key, 0)))
    return sorted(out, key=lambda r: r.key)


def test_concurrent_enrichment_matches_sequential():
    """Concurrent enrichment over multiple projects yields the SAME records as the sequential
    path (order-independent — both sorted by key)."""
    docs = _multi_project_docs()
    cc = _chunk_counts(docs)
    settings = dataclasses.replace(SETTINGS, enrich_concurrency=4)
    # use_llm=False → deterministic structural path, so the comparison is exact.
    concurrent = enrich_all(docs, cc, settings, use_llm=False)
    sequential = _sequential_records(docs, cc)
    assert [dataclasses.asdict(r) for r in concurrent] == \
        [dataclasses.asdict(r) for r in sequential]


def test_concurrent_enrichment_is_order_independent_and_complete():
    docs = _multi_project_docs()
    settings = dataclasses.replace(SETTINGS, enrich_concurrency=4)
    recs = enrich_all(docs, _chunk_counts(docs), settings, use_llm=False)
    keys = [r.key for r in recs]
    assert keys == sorted(keys)                       # deterministic order
    assert len(keys) == 4 and len(set(keys)) == 4     # one record per project, no dupes


class _FlakyLLM:
    """Fake LLM: returns valid JSON for most projects but RAISES for one project_id, to prove
    per-project failure isolation. No real backend; never spawns the `claude` CLI."""
    name = "fake"

    def __init__(self, fail_for: str):
        self.fail_for = fail_for

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        if f"/{self.fail_for}\n" in prompt:
            raise RuntimeError("simulated LLM failure")
        return '{"app_name": "X", "app_category": "tool", "theme_tags": ["t"], ' \
               '"has_humor": false, "one_line_summary": "ok"}'


def test_concurrent_failure_is_isolated(monkeypatch):
    """One project's LLM call failing must NOT kill the batch: the batch completes, every project
    gets a record, and the failing one degrades to structural-only (no LLM-derived summary)."""
    docs = _multi_project_docs()
    flaky = _FlakyLLM(fail_for="p2")
    monkeypatch.setattr("rageval.llm.get_llm", lambda settings=None: flaky)
    settings = dataclasses.replace(SETTINGS, enrich_concurrency=4)
    recs = enrich_all(docs, _chunk_counts(docs), settings, use_llm=True)
    by_id = {r.project_id: r for r in recs}
    assert set(by_id) == {"p1", "p2", "p3", "p4"}     # batch still complete
    # p2 failed → degraded: it got no LLM one_line_summary (structural-only path).
    assert by_id["p2"].one_line_summary is None
    # a non-failing project got the LLM summary.
    assert by_id["p1"].one_line_summary == "ok"


def test_sidecar_writes_happen_on_main_thread():
    """The on_record callback (the sidecar write) MUST be invoked on the calling/main thread, not a
    worker thread — that's what keeps SQLite single-threaded. We assert the thread identity."""
    docs = _multi_project_docs()
    main_thread = threading.get_ident()
    callback_threads: list[int] = []
    settings = dataclasses.replace(SETTINGS, enrich_concurrency=4)
    enrich_all(docs, _chunk_counts(docs), settings, use_llm=False,
               on_record=lambda rec: callback_threads.append(threading.get_ident()))
    assert callback_threads, "on_record was never called"
    assert all(t == main_thread for t in callback_threads)
