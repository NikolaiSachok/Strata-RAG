"""Tests for Tier-1 relevance classification and the dry-run manifest.

Classification is pure (SourceDoc + rules → Decision) so it's fully testable. We verify
the include/exclude rules fire with the RIGHT reason, and that the manifest's coverage
report correctly identifies blind spots over the sample corpus.
"""

from __future__ import annotations

from pathlib import Path

from rageval.classify import CorpusRules, classify, partition
from rageval.config import SAMPLE_CORPUS_DIR, SETTINGS
from rageval.manifest import build_manifest
from rageval.sources import discover_all
from rageval.sources.base import SourceDoc


def _doc(name: str, doc_type: str, text: str = "x" * 200, ext: str = "md",
         parts=("p",)) -> SourceDoc:
    return SourceDoc(
        project_id="0001", source_set="northwind",
        doc_path=Path(*parts) / name, doc_type=doc_type, ext=ext,
        raw_text=text, folder_meta={},
    )


RULES = CorpusRules.load()


def test_changelog_excluded_by_filename():
    dec = classify(_doc("CHANGELOG.md", "changelog"), RULES)
    assert not dec.include
    assert "changelog" in dec.reason


def test_promo_description_included():
    dec = classify(_doc("description.md", "promo"), RULES)
    assert dec.include
    assert dec.reason == "ok"


def test_noise_dir_excluded():
    dec = classify(_doc("shot.md", "other", parts=("project", "test", "screenshots")), RULES)
    assert not dec.include
    assert "noise dir" in dec.reason


def test_disallowed_extension_excluded():
    dec = classify(_doc("config.yaml", "other", ext="yaml"), RULES)
    assert not dec.include
    assert "ext not allowed" in dec.reason


def test_near_empty_excluded():
    dec = classify(_doc("description.md", "promo", text="hi"), RULES)
    assert not dec.include
    assert "near-empty" in dec.reason


def test_marker_doc_type_is_kept_despite_synthetic_ext_and_short_text():
    # A provenance 'marker' (Vega/Kotlin) has a synthetic ext and a short body — both of which
    # would normally drop it. keep_doc_types must short-circuit it to INCLUDE.
    dec = classify(_doc("1761", "marker", text="Project 1761 — non-conforming.", ext="marker"), RULES)
    assert dec.include
    assert "kept doc_type: marker" in dec.reason


def test_partition_splits_sample_corpus():
    docs = discover_all(SAMPLE_CORPUS_DIR)
    included, excluded = partition(docs, RULES)
    inc_types = {d.doc_type for d, _ in included}
    exc_reasons = {dec.reason for _, dec in excluded}
    # Real content survives; template/agent noise is dropped.
    assert "promo" in inc_types or "description" in inc_types
    assert any("changelog" in r for r in exc_reasons)


def test_manifest_flags_blind_spots():
    docs = discover_all(SAMPLE_CORPUS_DIR)
    manifest = build_manifest(docs, RULES, SETTINGS)
    blind = set(manifest.coverage.blind_spots)
    # atlas-relay ships only changelog + implementation_plan → must be a blind spot.
    assert "atlas/atlas-relay" in blind
    # atlas-quill ships only an engineering README → blind spot too.
    assert "atlas/atlas-quill" in blind
    # northwind/0001 has real content → must NOT be a blind spot.
    assert "northwind/0001" in set(manifest.coverage.projects_with_content)


# --- layout-audit rule additions -------------------------------------------

def test_prd_excluded_by_filename_and_glob():
    assert not classify(_doc("prd.md", "spec"), RULES).include
    glob_dec = classify(_doc("prd_v2.md", "spec"), RULES)
    assert not glob_dec.include
    assert "glob" in glob_dec.reason


def test_build_and_legal_docs_excluded():
    for fname in ("technical_guide.md", "assets_list.md", "setup.md",
                  "ATTRIBUTIONS.md", "Guidelines.md", "match3_game_spec.md"):
        dec = classify(_doc(fname, "spec"), RULES)
        assert not dec.include, f"{fname} should be excluded"


def test_coding_agent_build_docs_excluded_from_embedding():
    """design_system.md (exact filename) and *_interactive_mockup_requirements.md (glob) are
    raw coding-agent build instructions → excluded so they never flood top-k. The compact
    design_intent distillation that WOULD be embedded is a separate, ⏳-planned track."""
    ds = classify(_doc("design_system.md", "spec"), RULES)
    assert not ds.include and "design_system.md" in ds.reason
    mock = classify(_doc("orchard_interactive_mockup_requirements.md", "spec"), RULES)
    assert not mock.include and "glob" in mock.reason


def test_coding_agent_build_docs_excluded_end_to_end():
    """atlas-orchard ships a design_system.md + an *_interactive_mockup_requirements.md
    alongside a real description.docx — the build docs must be EXCLUDED, the description kept."""
    docs = discover_all(SAMPLE_CORPUS_DIR)
    included, excluded = partition(docs, RULES)
    orchard_inc = {d.doc_path.name for d, _ in included if d.project_id == "atlas-orchard"}
    orchard_exc = {d.doc_path.name for d, _ in excluded if d.project_id == "atlas-orchard"}
    assert "design_system.md" in orchard_exc
    assert "orchard_interactive_mockup_requirements.md" in orchard_exc
    # The real product description.docx is still included (not a blind spot).
    assert "description.docx" in orchard_inc


def test_docs_txt_config_excluded_but_promo_txt_kept():
    # docs/*.txt is tagged 'config' by the adapters → excluded as a noise doc_type...
    assert not classify(_doc("accounts.txt", "config", ext="txt"), RULES).include
    # ...while promo/*.txt is tagged 'promo' → kept (real store copy).
    assert classify(_doc("store_listing.txt", "promo", ext="txt"), RULES).include


def test_content_named_docs_txt_now_included_end_to_end():
    """False-negative fix: content-named docs/*.txt (now typed description/spec) survive
    classification, while a config-named one (typed config) is still dropped."""
    docs = discover_all(SAMPLE_CORPUS_DIR)
    included, excluded = partition(docs, RULES)
    inc_names = {d.doc_path.name for d, _ in included if d.project_id == "0001"}
    exc = {d.doc_path.name: dec.reason for d, dec in excluded if d.project_id == "0001"}
    # Real content docs/*.txt are now INCLUDED (previously dropped as 'config').
    assert "ideas.txt" in inc_names
    assert "design.txt" in inc_names
    # The genuine config dump is still EXCLUDED with the config doc_type reason.
    assert "setup.txt" in exc and "config" in exc["setup.txt"]


def test_settings_md_is_metadata_only_kept_but_not_embedded():
    # settings.md is tagged 'metadata' by the adapters → INCLUDED (enrich consumes it) but
    # flagged metadata_only so the indexer does NOT embed it.
    dec = classify(_doc("settings.md", "metadata"), RULES)
    assert dec.include and dec.metadata_only
    assert "metadata-only" in dec.reason
    assert dec.label == "ENRICH-ONLY"


def test_metadata_doc_not_embedded_but_present_for_enrich_in_manifest():
    """settings.md (atlas-ledger, doc_type 'metadata') must appear INCLUDED-but-enrich-only with
    ZERO embedded chunks, while still being in the manifest (so enrich can consume it)."""
    docs = discover_all(SAMPLE_CORPUS_DIR)
    manifest = build_manifest(docs, RULES, SETTINGS)
    settings_entry = next(e for e in manifest.included
                          if e.doc.doc_path.name == "settings.md")
    assert settings_entry.metadata_only is True
    assert settings_entry.n_chunks == 0          # NOT embedded
    assert settings_entry.n_redactions >= 3      # its keys are still scrubbed
    assert settings_entry.n_pii >= 1             # the internal owner email is PERSONAL PII


def test_published_contact_email_not_counted_as_pii_in_description():
    """A support@ email in a public promo/description is a PUBLISHED contact → NOT redacted, so
    it contributes 0 to the PII count (policy-aware, not blanket)."""
    docs = discover_all(SAMPLE_CORPUS_DIR)
    manifest = build_manifest(docs, RULES, SETTINGS)
    promo_entry = next(e for e in manifest.included
                       if e.doc.project_id == "0001" and e.doc.doc_type == "promo")
    assert promo_entry.n_pii == 0  # support@ in a description is kept


def test_sample_manifest_include_exclude_is_stable_after_phase4_decouple():
    """Phase-4 REGRESSION bar (#37): after moving allow_ext + filename policy behind the sample
    adapters' ClassificationPolicy, the sample corpus's include/exclude plan must be IDENTICAL —
    the classifier now resolves the per-corpus policy, but the outcome for the sample is unchanged.
    Pin the exact sets so a future decouple that shifts them fails loudly here."""
    docs = discover_all(SAMPLE_CORPUS_DIR)
    included, excluded = partition(docs, RULES)
    inc_ids = {d.doc_id for d, _ in included}
    # A representative slice of the byte-for-byte include set (per-corpus allow_ext = md/txt/docx +
    # the sample adapters' declared php/html/htm; docs/*.txt content-vs-config; settings.md meta).
    assert "atlas/atlas-vista/index.php" in inc_ids            # php allowed via adapter policy
    assert "northwind/0001/ideas.txt" in inc_ids               # content-named docs/*.txt kept
    assert "atlas/atlas-ledger/settings.md" in inc_ids         # metadata-only INCLUDED
    exc_ids = {d.doc_id for d, _ in excluded}
    assert "northwind/0001/setup.txt" in exc_ids               # config dump dropped
    assert any("changelog" in d.doc_path.name.lower() for d, _ in excluded)
    # The metadata-only flag survives the decouple.
    meta = [dec for d, dec in included if d.doc_path.name == "settings.md"]
    assert meta and all(dec.metadata_only for dec in meta)


def test_manifest_surfaces_redaction_count():
    docs = discover_all(SAMPLE_CORPUS_DIR)
    manifest = build_manifest(docs, RULES, SETTINGS)
    # The sample corpus has at least the settings.md secrets → a positive redaction count.
    assert manifest.total_redactions >= 3
    # And the settings.md entry specifically records its redactions.
    settings_entry = next(e for e in manifest.included
                          if e.doc.doc_path.name == "settings.md")
    assert settings_entry.n_redactions >= 3
