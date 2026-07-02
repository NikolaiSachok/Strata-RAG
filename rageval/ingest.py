"""Ingest: discover → CLASSIFY → (dry-run plan) → chunk → EMBED → Qdrant + sidecar.

This is the "offline" half of an enterprise RAG engine. Unlike a toy demo it does NOT
just read a flat folder of markdown — it runs the full inspectable pipeline:

  1. DISCOVER  — adapters (sources/) walk a heterogeneous corpus → SourceDocs (candidates).
  2. CLASSIFY  — corpus-rules.yaml decides INCLUDE/EXCLUDE per doc, relative to intent.
  3. DRY-RUN   — `--dry-run` prints the include/exclude + coverage MANIFEST and STOPS
                 (no embedding). Get discovery right before you pay to embed.
  4. CHUNK     — split included docs into overlapping windows (chunking.py).
  5. EMBED     — turn chunks into vectors (embeddings.py), batched.
  6. INDEX     — upsert vectors + payloads into Qdrant (index.py), HNSW.
  7. ENRICH    — an LLM pass produces structured per-project records (enrich.py) →
  8. SIDECAR   — persisted to SQLite for exact aggregation/intersection queries (sidecar.py).

The split between (1) discover and (2) classify is what makes the dry-run able to show
EXCLUDED files with reasons — the adapter found them; a rule dropped them.

    python -m rageval.ingest --dry-run     # plan only, no embedding
    python -m rageval.ingest               # full build (sample corpus by default)
    python -m rageval.ingest --no-enrich   # skip the LLM enrichment pass
    RAGENGINE_CORPUS_ROOT=/path/to/corpus python -m rageval.ingest   # real corpus
"""

from __future__ import annotations

import argparse

import dataclasses

from .chunking import Chunk, chunk_text_with_pages
from .classify import CorpusRules, partition
from .config import SETTINGS, Settings, is_sample_corpus
from .embeddings import get_embedder
from .manifest import build_manifest, render_manifest
from .pii import get_pii_detector
from .redact import PiiPolicy, redact
from .sidecar import connect, upsert_project
from .sources import discover_all
from .sources.base import SourceDoc


def redact_included(included_docs: list[SourceDoc],
                    pii_policy: PiiPolicy = PiiPolicy()) -> tuple[list[SourceDoc], int, int]:
    """Scrub secret VALUES and PERSONAL PII from every included doc's text, AFTER extraction
    and BEFORE chunking. Defense-in-depth: even a doc we deliberately KEEP (e.g. settings.md,
    kept for its Brand field) gets its keys AND any personal email redacted here so no
    credential or personal data is ever embedded, stored in a chunk payload, or retrievable.

    Two DISTINCT guardrail categories run here (secrets then policy-aware PII — see
    redact.redact). The PII pass is CONTEXT-AWARE: it gets each doc's `doc_type` (provenance)
    and the configured `pii_policy`, so a published support@ contact in a description is KEPT
    while a personal email in an internal settings/pitch doc is redacted. Returns
    (redacted_docs, total_secrets, total_pii). SourceDoc is frozen, so we rebuild each with the
    cleaned text via dataclasses.replace — keeping the rest of its provenance intact.
    """
    out: list[SourceDoc] = []
    total_secrets = 0
    total_pii = 0
    # Build the PII DETECTOR once (regex DEFAULT, or presidio via RAGEVAL_PII_BACKEND) and reuse
    # it across docs — constructing the presidio detector loads a spaCy model, so per-doc
    # construction would be ruinous. The regex detector is cheap either way.
    detector = get_pii_detector()
    for d in included_docs:
        clean, n_sec, n_pii = redact(d.raw_text, doc_type=d.doc_type, pii_policy=pii_policy,
                                     detector=detector)
        total_secrets += n_sec
        total_pii += n_pii
        out.append(dataclasses.replace(d, raw_text=clean) if (n_sec or n_pii) else d)
    return out, total_secrets, total_pii


# The synthetic-sample source-set families (the fictional corpora that ship in data/sample/).
# A source_set's family is the part before the first '-' (mirrors roster._tsv_stem_for):
# "atlas-ledger" → "atlas". Any family NOT in this set is treated as a custom corpus. This is
# the same sample-vs-custom split the collection-name suffix encodes.
_SAMPLE_FAMILIES = {"northwind", "atlas"}


def _is_sample_source_set(source_set: str) -> bool:
    return source_set.split("-", 1)[0].lower() in _SAMPLE_FAMILIES


def _warn_on_corpus_mismatch(existing: set[str], *, ingesting_sample: bool) -> None:
    """DEFENSE-IN-DEPTH (the naming fix is the primary fix): if the target collection already
    holds points from the OTHER corpus family than the one we're about to ingest, print a warning.

    With the `_sample` suffix this can normally only happen via an explicit RAGEVAL_COLLECTION
    override that points a sample ingest at a real collection (or vice versa) — exactly the
    cross-corpus contamination the suffix otherwise prevents. Warn, don't block: the human chose
    the override, but a contaminated index is how test data once leaked into a real answer's
    citations, so it must be visible."""
    from .enrich import _log

    foreign = {s for s in existing if _is_sample_source_set(s) != ingesting_sample}
    if foreign:
        kind = "real-corpus" if ingesting_sample else "sample"
        sample_of = ", ".join(sorted(foreign)[:5])
        _log(f"⚠ [contamination] target collection already holds {kind} points "
             f"(source_set: {sample_of}) but you're ingesting "
             f"{'the sample' if ingesting_sample else 'a real'} corpus — "
             f"check RAGEVAL_COLLECTION / corpus root to avoid mixing corpora in one index.")


def build_chunks(included_docs: list[SourceDoc], settings: Settings = SETTINGS) -> list[Chunk]:
    """Chunk every INCLUDED doc into retrievable units, carrying provenance metadata.

    PDF page provenance (MAJOR-1): chunk with `chunk_text_with_pages` so EVERY chunk resolves to the
    page its content STARTS on (not just a page's first chunk) — the page rides into the payload and
    out to the citation. Non-PDF docs have no `[page N]` markers, so `page` is None."""
    chunks: list[Chunk] = []
    for d in included_docs:
        for i, (piece, page) in enumerate(chunk_text_with_pages(
                d.raw_text, size=settings.chunk_size, overlap=settings.chunk_overlap)):
            chunks.append(
                Chunk(
                    text=piece,
                    project_id=d.project_id,
                    source_set=d.source_set,
                    source=d.doc_path.name,
                    doc_type=d.doc_type,
                    chunk_index=i,
                    page=page,
                )
            )
    return chunks


def ingest(settings: Settings = SETTINGS, *, use_enrich: bool = True,
           recreate: bool = False) -> dict:
    """Run the full offline pipeline. Returns a small summary dict.

    Imports of qdrant_client/index are LOCAL to this function so that `--dry-run` (and
    tests of the planning logic) don't require a running Qdrant.
    """
    from .enrich import _log
    from .index import (count, ensure_collection, existing_source_sets, get_client,
                        upsert_chunks)

    rules = CorpusRules.load()
    all_docs = discover_all(settings.corpus_root)
    included_pairs, _excluded = partition(all_docs, rules)
    _log(f"[discovery] {len(included_pairs)} docs included / "
         f"{len(all_docs) - len(included_pairs)} excluded (of {len(all_docs)} discovered)")
    # Two buckets out of the included set: docs to EMBED (real narrative content) and docs that
    # are METADATA-ONLY (settings.md → enrich consumes them, the indexer skips them). Everything
    # included is enriched; only the non-metadata_only docs are chunked + embedded.
    included = [d for d, _ in included_pairs]                                   # all enriched
    embeddable = [d for d, dec in included_pairs if not dec.metadata_only]      # only these embed

    # REDACT secrets AND PII after classification, before chunking. From here on, no downstream
    # stage (embed / payload / enrich) ever sees a live credential or a personal email. The PII
    # pass uses the policy the human set in corpus-rules.yaml (published contacts preserved).
    included, n_redactions, n_pii = redact_included(included, rules.pii_policy)
    # Re-derive the embeddable subset from the redacted docs (same doc_ids, now scrubbed).
    embeddable_ids = {d.doc_id for d in embeddable}
    embeddable = [d for d in included if d.doc_id in embeddable_ids]

    chunks = build_chunks(embeddable, settings)

    # EMBED (batched — far more efficient than one call per chunk).
    embedder = get_embedder(settings)
    vectors = embedder.embed([c.text for c in chunks]) if chunks else []
    _log(f"[embedding] embedded {len(chunks)} chunks "
         f"({settings.embeddings}/{settings.embed_model})")

    # INDEX into Qdrant (the collection name is model-derived + `_sample`-suffixed for the sample
    # corpus — see settings.collection_name — so the sample can't share a real collection).
    client = get_client(settings)
    ensure_collection(client, settings, recreate=recreate)
    # Defense-in-depth: warn if the target collection already holds the OTHER corpus family
    # (only reachable via a RAGEVAL_COLLECTION override now that naming isolates the sample).
    _warn_on_corpus_mismatch(existing_source_sets(client, settings),
                             ingesting_sample=is_sample_corpus(settings.corpus_root))
    n_chunks = upsert_chunks(client, chunks, vectors, settings)
    _log(f"[upsert] upserted {n_chunks} chunks into {settings.collection_name}")

    # ENRICH → SIDECAR. chunk_counts let the sidecar flag chunk-count outliers later.
    chunk_counts: dict[tuple[str, str], int] = {}
    for c in chunks:
        chunk_counts[(c.source_set, c.project_id)] = chunk_counts.get(
            (c.source_set, c.project_id), 0) + 1

    from .enrich import enrich_all

    # SIDECAR WRITES ON THE MAIN THREAD: enrich_all runs the LLM calls on worker threads but
    # invokes on_record HERE, in the calling thread, as each future completes — so this single
    # SQLite connection is only ever touched by one thread (no 'database is locked').
    conn = connect()
    records = enrich_all(included, chunk_counts, settings, use_llm=use_enrich,
                         on_record=lambda rec: upsert_project(conn, rec))
    # STRUCTURED, facts-only entities (#41 — spreadsheet rows etc.): a SEPARATE path from the
    # document enrich loop. Each row becomes its own facts-only sidecar record (chunk_count 0, never
    # embedded), so an aggregation question over the spreadsheet answers from the sidecar via the
    # existing `aggregate` path — with NO raw-table dilution of the vector index.
    n_entities = _ingest_structured_entities(conn, settings)
    conn.close()

    return {
        "included_docs": len(included),
        "excluded_docs": len(all_docs) - len(included),
        "chunks_indexed": n_chunks,
        "secrets_redacted": n_redactions,
        "pii_redacted": n_pii,
        "projects_enriched": len(records),
        "structured_entities": n_entities,
        "total_in_collection": count(client, settings),
        "collection": settings.collection_name,
    }


def _ingest_structured_entities(conn, settings: Settings = SETTINGS) -> int:
    """Harvest STRUCTURED, facts-only entities (#41) from every adapter's `harvest_entities` hook
    and write one facts-only sidecar record per entity. Returns the number written.

    Corpus-agnostic: the adapters own the column→facet mapping (the core knows no column names);
    here we just fold the yielded facts into records (fail-closed against declared facets) and
    upsert. A corpus with no tabular data yields nothing → a clean no-op. Runs on the MAIN thread
    (same single SQLite connection as the document enrich writes)."""
    from .enrich import _log, entities_to_records
    from .sources import harvest_all_entities

    entities = harvest_all_entities(settings.corpus_root)
    if not entities:
        return 0
    records = entities_to_records(entities)
    for rec in records:
        upsert_project(conn, rec)
    _log(f"[structured] {len(records)} facts-only entities (rows) written to the sidecar "
         f"(NOT embedded — queryable via aggregate)")
    return len(records)


def enrich_only(settings: Settings = SETTINGS, *, use_enrich: bool = True) -> dict:
    """Re-run ONLY enrichment (LLM + config.yaml harvest + roster join) and rewrite the
    sidecar — WITHOUT re-embedding or re-upserting. The Qdrant index is left completely untouched.

    WHY: a metadata-schema change (e.g. splitting a product-name field into app_name + publisher)
    needs the
    sidecar rebuilt, but NOT the ~12k-chunk vector index re-embedded — embeddings don't change.
    This is the cheap refresh path: discover → classify → redact → chunk (for chunk_counts only,
    nothing is upserted) → enrich → sidecar. No embedder, no Qdrant client, no upsert.
    """
    from .enrich import _log, enrich_all

    rules = CorpusRules.load()
    all_docs = discover_all(settings.corpus_root)
    included_pairs, _excluded = partition(all_docs, rules)
    included = [d for d, _ in included_pairs]
    embeddable = [d for d, dec in included_pairs if not dec.metadata_only]
    _log(f"[enrich-only] {len(included)} docs included; refreshing sidecar (NO re-embed/upsert)")

    # Redact (so any newly-derived structured field still sees scrubbed text), then chunk JUST to
    # recover per-project chunk_counts for the sidecar outlier/blind-spot flags. Nothing is embedded.
    included, _n_sec, _n_pii = redact_included(included, rules.pii_policy)
    embeddable_ids = {d.doc_id for d in embeddable}
    embeddable = [d for d in included if d.doc_id in embeddable_ids]
    chunks = build_chunks(embeddable, settings)
    chunk_counts: dict[tuple[str, str], int] = {}
    for c in chunks:
        chunk_counts[(c.source_set, c.project_id)] = chunk_counts.get(
            (c.source_set, c.project_id), 0) + 1

    conn = connect()
    records = enrich_all(included, chunk_counts, settings, use_llm=use_enrich,
                         on_record=lambda rec: upsert_project(conn, rec))
    # Refresh the structured facts-only entities too (spreadsheet rows), so a sidecar refresh is
    # complete without re-embedding.
    n_entities = _ingest_structured_entities(conn, settings)
    conn.close()
    return {
        "included_docs": len(included),
        "projects_enriched": len(records),
        "chunks_counted": len(chunks),
        "structured_entities": n_entities,
    }


def dry_run(settings: Settings = SETTINGS) -> str:
    """Build and render the manifest WITHOUT embedding or touching Qdrant."""
    rules = CorpusRules.load()
    all_docs = discover_all(settings.corpus_root)
    manifest = build_manifest(all_docs, rules, settings)
    return render_manifest(manifest, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a corpus into Qdrant + the metadata sidecar.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the include/exclude + coverage manifest and stop (no embedding).")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip the LLM metadata-enrichment pass (sidecar gets structural fields only).")
    parser.add_argument("--enrich-only", action="store_true",
                        help="Re-run ONLY enrichment (LLM + config.yaml harvest + roster join) and "
                             "rewrite the sidecar, WITHOUT re-embedding or re-upserting (Qdrant untouched). "
                             "Use after a metadata-schema change to avoid a full re-embed.")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate the Qdrant collection before ingesting.")
    args = parser.parse_args()

    settings = Settings.load()

    if args.dry_run:
        print(dry_run(settings))
        print("\n(dry-run: nothing was embedded. Re-run without --dry-run to build the index.)")
        return

    if args.enrich_only:
        print(f"Re-enriching (sidecar refresh, NO re-embed) from {settings.corpus_root} ...")
        summary = enrich_only(settings, use_enrich=not args.no_enrich)
        print(
            "Enrich-only complete (Qdrant index untouched):\n"
            f"  included docs     : {summary['included_docs']}\n"
            f"  chunks counted    : {summary['chunks_counted']}\n"
            f"  projects enriched : {summary['projects_enriched']}\n"
            "Now run:  python -m rageval.inspect --sidecar"
        )
        return

    print(f"Ingesting corpus from {settings.corpus_root} ...")
    summary = ingest(settings, use_enrich=not args.no_enrich, recreate=args.recreate)
    print(
        "Ingest complete:\n"
        f"  included docs     : {summary['included_docs']}\n"
        f"  excluded docs     : {summary['excluded_docs']}\n"
        f"  chunks indexed    : {summary['chunks_indexed']}\n"
        f"  secrets redacted  : {summary['secrets_redacted']}\n"
        f"  PII redacted      : {summary['pii_redacted']}\n"
        f"  projects enriched : {summary['projects_enriched']}\n"
        f"  structured rows   : {summary['structured_entities']}\n"
        f"  total in Qdrant   : {summary['total_in_collection']}\n"
        f"  embeddings        : {settings.embeddings} ({settings.embed_model})\n"
        f"  collection        : {summary['collection']}\n"
        "Now run:  python -m rageval.inspect            (browse chunks / coverage)\n"
        "      or:  python -m rageval.eval              (retrieval + faithfulness metrics)"
    )


if __name__ == "__main__":
    main()
