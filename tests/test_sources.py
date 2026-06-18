"""Tests for source-adapter DISCOVERY and .docx PARSING against the sample corpus.

These run over the committed, fictional data/sample/ corpus, so they're deterministic and
need no model or network. They lock down the two things adapters must get right: finding
the candidate docs, and extracting text from legacy .docx files.
"""

from __future__ import annotations

import pytest

from rageval.config import SAMPLE_CORPUS_DIR
from rageval.sources import discover_all
from rageval.sources.base import (
    SourceAdapter,
    SourceDoc,
    docs_txt_doc_type,
    is_store_listing_txt,
)
from rageval.sources.atlas import AtlasAdapter, strip_markup
from rageval.sources.northwind import NorthwindAdapter


def test_discover_all_finds_every_source_set():
    docs = discover_all(SAMPLE_CORPUS_DIR)
    source_sets = {d.source_set for d in docs}
    # The sample corpus exercises both bundled adapter shapes (northwind/atlas).
    assert {"northwind", "atlas"} <= source_sets


def test_northwind_finds_promo_and_changelog_candidates():
    adapter = NorthwindAdapter(SAMPLE_CORPUS_DIR / "northwind")
    docs = list(adapter.discover())
    by_id = {d.doc_id: d for d in docs}
    # The promo description is discovered AND typed as promo content...
    promo = [d for d in docs if d.doc_type == "promo"]
    assert any(d.project_id == "0001" for d in promo)
    # ...and the noise changelog is ALSO discovered (discovery != classification).
    assert any(d.doc_type == "changelog" for d in docs)
    # project 0004 has only a pubspec.yaml → no text candidates discovered for it.
    assert not any(d.project_id == "0004" for d in docs)


def test_atlas_parses_docx_content():
    adapter = AtlasAdapter(SAMPLE_CORPUS_DIR / "atlas")
    docs = list(adapter.discover())
    docx_docs = [d for d in docs if d.ext == "docx"]
    assert docx_docs, "expected at least one .docx in the atlas sample corpus"
    ledger = [d for d in docx_docs if d.project_id == "atlas-ledger"]
    assert ledger, "atlas-ledger should have a parsed .docx"
    # The parsed text must contain real content from the Word file.
    assert "lemon" in ledger[0].raw_text.lower()
    assert "budgeting" in ledger[0].raw_text.lower()


def test_sourcedoc_identity_is_stable():
    docs = discover_all(SAMPLE_CORPUS_DIR)
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids))  # all doc ids unique


# --- path-aware .txt typing (layout-audit finding) --------------------------

def test_docs_txt_tagged_config_and_promo_txt_tagged_promo():
    docs = list(AtlasAdapter(SAMPLE_CORPUS_DIR / "atlas").discover())
    by_name = {d.doc_path.name: d for d in docs}
    # docs/accounts.txt → config (credential dump).
    assert by_name["accounts.txt"].doc_type == "config"
    # A promo/*.txt store listing that IS yielded (the store-only atlas-summit) → promo copy.
    summit = [d for d in docs if d.project_id == "atlas-summit"]
    assert summit and summit[0].doc_type == "promo"


# --- docs/*.txt content-vs-config disambiguation (false-negative fix) -------

def test_docs_txt_doc_type_unit():
    # Content-named docs/*.txt → real content types (INCLUDE)...
    assert docs_txt_doc_type("description.txt") == "description"
    assert docs_txt_doc_type("ideas.txt") == "spec"
    assert docs_txt_doc_type("design.txt") == "spec"
    # ...genuine config/credential dumps → 'config' (kept excluded).
    assert docs_txt_doc_type("accounts.txt") == "config"
    assert docs_txt_doc_type("settings.txt") == "config"
    assert docs_txt_doc_type("setup.txt") == "config"
    # An unknown docs/*.txt defaults to 'config' (conservative).
    assert docs_txt_doc_type("mystery.txt") == "config"


def test_content_named_docs_txt_typed_as_content_and_config_named_excluded():
    """northwind/0001 ships content-named docs/*.txt (ideas.txt → spec, design.txt → spec)
    that must be KEPT, and a config-named setup.txt that must stay 'config' (excluded)."""
    docs = list(NorthwindAdapter(SAMPLE_CORPUS_DIR / "northwind").discover())
    by_name = {d.doc_path.name: d for d in docs if d.project_id == "0001"}
    assert by_name["ideas.txt"].doc_type == "spec"
    assert by_name["design.txt"].doc_type == "spec"
    assert by_name["setup.txt"].doc_type == "config"  # credential dump stays config


# --- index.php / index.html conditional promo fallback ----------------------

def test_strip_markup_recovers_visible_text_only():
    raw = (
        "<?php $secret = 'x'; ?>"
        "<html><head><style>.h{}</style><script>evil()</script></head>"
        "<body><h1>Vista Weather</h1><p>Mountain&mdash;themed forecasts.</p></body></html>"
    )
    visible = strip_markup(raw)
    assert "Vista Weather" in visible
    assert "Mountain—themed forecasts." in visible  # entity unescaped
    assert "secret" not in visible and "evil" not in visible and "<" not in visible


def test_index_fallback_fires_only_when_no_description():
    docs = list(AtlasAdapter(SAMPLE_CORPUS_DIR / "atlas").discover())
    # atlas-vista has ONLY a back/index.php → the fallback yields its promo copy.
    vista = [d for d in docs if d.project_id == "atlas-vista"]
    assert vista, "index.php-only project should produce a fallback promo doc"
    assert vista[0].doc_type == "promo"
    assert "vista weather" in vista[0].raw_text.lower()
    # A project WITH a description (atlas-ledger has description.docx) must NOT emit an
    # index.* fallback even if one existed → no php/html docs for it.
    ledger_index = [d for d in docs if d.project_id == "atlas-ledger"
                    and d.ext in ("php", "html", "htm")]
    assert ledger_index == []


# --- provenance-aware dedup of DERIVED store listings (promo fallback) -------

def test_store_listing_suppressed_when_canonical_description_exists():
    """atlas-ledger has a canonical description.docx PLUS two DERIVED store listings
    (description_app_store.txt + description_google_play.txt). Only the canonical must be
    yielded; the near-duplicate store-txt files must be SUPPRESSED (not in retrieval)."""
    docs = list(AtlasAdapter(SAMPLE_CORPUS_DIR / "atlas").discover())
    ledger = [d for d in docs if d.project_id == "atlas-ledger"]
    names = {d.doc_path.name for d in ledger}
    # The canonical description survives...
    assert "description.docx" in names
    # ...and BOTH derived store listings are gone (provenance-aware dedup).
    assert "description_app_store.txt" not in names
    assert "description_google_play.txt" not in names
    # Sanity: no store-listing txt of ANY kind leaked through for this project.
    assert not any(is_store_listing_txt(d.doc_path.name) for d in ledger)


def test_store_listing_fallback_yields_when_no_canonical_description():
    """atlas-summit is a STORE-ONLY project: its only product copy is an app-store listing,
    with no description.md/.docx. The fallback must yield it (zero blind spots) as promo."""
    docs = list(AtlasAdapter(SAMPLE_CORPUS_DIR / "atlas").discover())
    summit = [d for d in docs if d.project_id == "atlas-summit"]
    assert summit, "store-only project must still produce content via the fallback"
    assert summit[0].doc_path.name == "description_app_store.txt"
    assert summit[0].doc_type == "promo"
    assert "summit stepper" in summit[0].raw_text.lower()


def test_store_listing_detector_recognises_known_variants():
    assert is_store_listing_txt("description_app_store.txt")
    assert is_store_listing_txt("description_google_play.txt")
    assert is_store_listing_txt("store_listing.txt")
    assert is_store_listing_txt("en_app_store.txt")
    # Canonical description and arbitrary copy are NOT store listings.
    assert not is_store_listing_txt("description.md")
    assert not is_store_listing_txt("description.txt")
    assert not is_store_listing_txt("overview.txt")


# --- open-core registry extension seam --------------------------------------
# These lock down the public/private split: the CORE registers only the sample adapters;
# private adapters arrive via the optional bootstrap; and register_adapter() is the seam.

from rageval.sources import registry  # noqa: E402


@pytest.fixture
def clean_registry():
    """Snapshot and restore the global ADAPTER_BY_FOLDER so a test can mutate it freely."""
    saved = dict(registry.ADAPTER_BY_FOLDER)
    try:
        yield
    finally:
        registry.ADAPTER_BY_FOLDER.clear()
        registry.ADAPTER_BY_FOLDER.update(saved)


def test_core_registry_is_sample_only_without_private_bootstrap(clean_registry, monkeypatch):
    """With no optional bootstrap loaded, re-running the core's registration yields ONLY the
    bundled sample adapters (northwind + atlas) — no error. This is the property the engine
    must have for free when no out-of-tree adapters are present."""
    # Reset to the core defaults, then make the optional bootstrap a no-op (as if
    # _private_plugins.py were absent).
    registry.ADAPTER_BY_FOLDER.clear()
    registry.ADAPTER_BY_FOLDER.update({
        "northwind": NorthwindAdapter,
        "atlas": AtlasAdapter,
    })
    monkeypatch.setattr(registry, "_load_private_plugins", lambda: None)
    registry._load_private_plugins()  # the default path: import absent → no-op

    assert set(registry.ADAPTER_BY_FOLDER) == {"northwind", "atlas"}


def test_register_adapter_adds_and_dispatches(clean_registry, tmp_path):
    """register_adapter() (the public extension API) wires a new folder→adapter mapping and
    get_adapters/discover_all then dispatch a real corpus folder to it."""
    class WidgetAdapter(SourceAdapter):
        source_set = "widget"

        def discover(self):
            yield SourceDoc(
                project_id="w1", source_set=self.source_set,
                doc_path=self.root / "w1" / "description.md", doc_type="description",
                ext="md", raw_text="a fictional widget product description",
            )

    registry.register_adapter("widget", WidgetAdapter)
    assert registry.ADAPTER_BY_FOLDER["widget"] is WidgetAdapter

    # Lay out a corpus root with a matching folder and confirm dispatch.
    (tmp_path / "widget").mkdir()
    adapters = registry.get_adapters(tmp_path)
    assert any(isinstance(a, WidgetAdapter) for a in adapters)
    docs = registry.discover_all(tmp_path)
    assert [d.source_set for d in docs] == ["widget"]


def test_register_adapter_rejects_non_adapter(clean_registry):
    """The seam guards against mis-wiring: a non-SourceAdapter value is rejected."""
    with pytest.raises(TypeError):
        registry.register_adapter("bogus", object)  # type: ignore[arg-type]
