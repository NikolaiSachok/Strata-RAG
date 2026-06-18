"""Tests for the embedding-A/B plumbing: model-derived collection names + kind-filtered golden.

These are pure/deterministic (no Qdrant): the collection name is derived from settings, and the
golden loader filters by `kind`. Both are the load-bearing pieces of the MiniLM-vs-mpnet A/B —
if two models shared a collection, or if brand/metadata questions leaked into the retrieval
metric, the comparison would be meaningless.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from rageval.config import COLLECTION_BASE, SAMPLE_COLLECTION_SUFFIX, Settings, slug

# A fictional REAL-corpus root (NOT under data/sample/) — exercises the unchanged real-corpus name.
_REAL_ROOT = Path("/tmp/some-real-corpus-root")


def _real(**kw) -> Settings:
    """Settings whose corpus_root is a (fictional) real corpus, so collection_name carries NO
    sample suffix. The shipped default corpus_root IS the sample, which would suffix the name.
    Defaults collection_override to "" (model-derivation active) unless a caller overrides it."""
    kw.setdefault("collection_override", "")
    return dataclasses.replace(Settings.load(), corpus_root=_REAL_ROOT, **kw)


def test_slug_normalises_model_ids():
    assert slug("all-MiniLM-L6-v2") == "all_minilm_l6_v2"
    assert slug("sentence-transformers/all-mpnet-base-v2") == "sentence_transformers_all_mpnet_base_v2"
    assert slug("Weird__Name!!") == "weird_name"


def test_collection_name_is_model_derived_and_distinct():
    # Real corpus → name is the bare model-derived name (no sample suffix).
    mini = _real(embed_model="all-MiniLM-L6-v2", embed_dim=384)
    mpnet = _real(embed_model="all-mpnet-base-v2", embed_dim=768)
    assert mini.collection_name == f"{COLLECTION_BASE}_all_minilm_l6_v2"
    assert mpnet.collection_name == f"{COLLECTION_BASE}_all_mpnet_base_v2"
    # The whole point of the A/B: the two models do NOT share an index.
    assert mini.collection_name != mpnet.collection_name


def test_real_corpus_name_is_unchanged_no_suffix():
    # A real-corpus ingest keeps the legacy model-derived name — the existing real index stays
    # valid (no forced re-ingest). No `_sample` anywhere.
    s = _real(embed_model="all-mpnet-base-v2", embed_dim=768)
    assert s.collection_name == f"{COLLECTION_BASE}_all_mpnet_base_v2"
    assert SAMPLE_COLLECTION_SUFFIX not in s.collection_name


def test_sample_corpus_gets_its_own_suffixed_collection():
    # The default corpus_root IS the synthetic sample → the name is suffixed, so a sample ingest
    # can NEVER upsert into the real collection above.
    s = dataclasses.replace(Settings.load(), embed_model="all-mpnet-base-v2", embed_dim=768,
                            collection_override="")
    assert s.collection_name == f"{COLLECTION_BASE}_all_mpnet_base_v2_{SAMPLE_COLLECTION_SUFFIX}"


def test_sample_subdirectory_also_counts_as_sample():
    # A path UNDER data/sample/ (not just exactly it) is still the sample corpus.
    from rageval.config import SAMPLE_CORPUS_DIR

    s = dataclasses.replace(Settings.load(), corpus_root=SAMPLE_CORPUS_DIR / "northwind",
                            embed_model="all-mpnet-base-v2", embed_dim=768, collection_override="")
    assert s.collection_name.endswith(f"_{SAMPLE_COLLECTION_SUFFIX}")


def test_sample_and_real_collections_differ_for_same_model():
    # Same embedding model, different corpus → DIFFERENT collections. This is the contamination fix:
    # the sample can't land in the real model's collection.
    real = _real(embed_model="all-mpnet-base-v2", embed_dim=768)
    sample = dataclasses.replace(Settings.load(), embed_model="all-mpnet-base-v2", embed_dim=768,
                                 collection_override="")
    assert real.collection_name != sample.collection_name


def test_model_still_differentiates_within_sample():
    # The A/B axis survives isolation: two models over the sample still get distinct collections.
    mini = dataclasses.replace(Settings.load(), embed_model="all-MiniLM-L6-v2", embed_dim=384,
                               collection_override="")
    mpnet = dataclasses.replace(Settings.load(), embed_model="all-mpnet-base-v2", embed_dim=768,
                                collection_override="")
    assert mini.collection_name != mpnet.collection_name
    assert mini.collection_name.endswith(f"_{SAMPLE_COLLECTION_SUFFIX}")
    assert mpnet.collection_name.endswith(f"_{SAMPLE_COLLECTION_SUFFIX}")


def test_collection_override_wins():
    # Explicit override beats BOTH the model-derivation AND the sample suffix — on sample or real.
    on_sample = dataclasses.replace(Settings.load(), collection_override="custom_ab_run")
    on_real = _real(collection_override="custom_ab_run")
    assert on_sample.collection_name == "custom_ab_run"
    assert on_real.collection_name == "custom_ab_run"


def test_contamination_guard_classifies_source_set_families():
    # The ingest defense-in-depth guard: sample families (northwind/atlas, incl. -subset variants)
    # vs custom families. Mirrors the sample-vs-custom split the suffix encodes.
    from rageval.ingest import _is_sample_source_set

    assert _is_sample_source_set("northwind")
    assert _is_sample_source_set("atlas-ledger")
    assert not _is_sample_source_set("custom")
    assert not _is_sample_source_set("custom-extra")
    assert not _is_sample_source_set("other")


def test_contamination_warning_fires_only_on_cross_corpus_mix(capsys):
    # Warn when ingesting the sample but the collection holds custom-corpus points (and vice
    # versa); stay silent when families match.
    from rageval.ingest import _warn_on_corpus_mismatch

    _warn_on_corpus_mismatch({"custom", "other"}, ingesting_sample=True)
    assert "contamination" in capsys.readouterr().out
    _warn_on_corpus_mismatch({"northwind", "atlas"}, ingesting_sample=False)
    assert "contamination" in capsys.readouterr().out
    _warn_on_corpus_mismatch({"northwind", "atlas"}, ingesting_sample=True)
    assert capsys.readouterr().out == ""        # same family → no warning
    _warn_on_corpus_mismatch(set(), ingesting_sample=True)
    assert capsys.readouterr().out == ""        # empty collection (first ingest) → no warning


def test_golden_kind_filter_defaults_missing_to_retrieval(tmp_path):
    from rageval.eval import load_golden

    p = tmp_path / "g.yaml"
    p.write_text(
        "questions:\n"
        "  - id: q_theme\n"
        "    question: fruity theme?\n"
        "    relevant: ['a/1']\n"                       # no kind → defaults to retrieval
        "  - id: q_brand\n"
        "    kind: metadata\n"
        "    question: which promote BrandX?\n"
        "    relevant: ['a/2']\n",
        encoding="utf-8",
    )
    retrieval = load_golden(p, kind="retrieval")
    metadata = load_golden(p, kind="metadata")
    everything = load_golden(p, kind="all")
    assert [q["id"] for q in retrieval] == ["q_theme"]   # the kind-less one defaults in
    assert [q["id"] for q in metadata] == ["q_brand"]
    assert {q["id"] for q in everything} == {"q_theme", "q_brand"}
    # The A/B metric must NOT include the brand/metadata question.
    assert "q_brand" not in {q["id"] for q in retrieval}


def test_sample_golden_is_all_retrieval():
    # The shipped sample golden has no kinds → every question is retrieval (the full A/B set).
    from rageval.eval import load_golden

    assert load_golden(kind="metadata") == []
    assert len(load_golden(kind="retrieval")) == len(load_golden(kind="all"))
