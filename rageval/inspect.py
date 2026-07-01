"""Inspect: browse what ACTUALLY became embeddings, and audit coverage/quality.

After ingest, you want to EYEBALL the index before trusting it: did a .docx get
truncated? Did template boilerplate sneak in? Is some project a blind spot? This module
is the post-ingest half of the observability story (manifest.py is the pre-ingest half).

Subcommands:
  python -m rageval.inspect --project atlas-ledger   # browse one project's chunks
  python -m rageval.inspect --coverage               # blind spots + chunk-count outliers
  python -m rageval.inspect --sidecar                # dump the structured metadata table
  python -m rageval.inspect --audit                  # quality flags (near-empty, code-ish, dup)

Everything here reads back from Qdrant (the chunk payloads) and the SQLite sidecar —
the same data the retriever uses — so what you see is exactly what the engine sees.
"""

from __future__ import annotations

import argparse
import statistics

from .config import SETTINGS, Settings
from .sidecar import all_projects, connect


def _hits_for(project: str | None):
    from .config import Settings
    from .index import get_client, scroll_all

    settings = Settings.load()  # resolves the model-derived (or overridden) collection
    client = get_client(settings)
    return list(scroll_all(client, project_id=project, settings=settings))


def browse_project(project: str) -> str:
    payloads = _hits_for(project)
    payloads.sort(key=lambda p: (str(p.get("source", "")), int(p.get("chunk_index", 0))))
    lines = [f"CHUNKS for project '{project}' — {len(payloads)} chunk(s)"]
    if not payloads:
        lines.append("  (none — is the id correct? did ingest run? is it a blind spot?)")
    for pl in payloads:
        text = " ".join(str(pl.get("text", "")).split())
        lines.append(
            f"  [{pl.get('source')}::{pl.get('chunk_index')}] doc_type={pl.get('doc_type')}\n"
            f"      {text[:220]}{'…' if len(text) > 220 else ''}"
        )
    return "\n".join(lines)


def coverage_report() -> str:
    """Blind spots + chunk-count outliers, computed from the sidecar."""
    conn = connect()
    recs = all_projects(conn)
    conn.close()
    lines = ["COVERAGE (from the sidecar)"]
    blind = [r.key for r in recs if r.chunk_count == 0]
    lines.append(f"  projects in sidecar : {len(recs)}")
    lines.append(f"  BLIND SPOTS (0 chunks): {len(blind)}")
    for k in sorted(blind):
        lines.append(f"      !!  {k}")
    counts = [r.chunk_count for r in recs if r.chunk_count > 0]
    if len(counts) >= 3:
        med = statistics.median(counts)
        outliers = [r for r in recs if r.chunk_count >= max(3 * med, med + 2)]
        lines.append(f"  chunk-count outliers (>= 3x median={med}): {len(outliers)}")
        for r in sorted(outliers, key=lambda x: -x.chunk_count):
            lines.append(f"      ??  {r.key}  ({r.chunk_count} chunks)")
    # THIN-METADATA flag (observability for the enrich step): projects where enrichment produced
    # low-confidence metadata (no structured source, no brand/category) — a reviewer can see WHERE
    # the metadata is weak and decide whether to add a settings.md or tune extraction.
    thin = [r for r in recs if r.metadata_confidence == "low"]
    lines.append(f"  THIN metadata (low enrich confidence): {len(thin)}")
    for r in sorted(thin, key=lambda x: x.key):
        lines.append(f"      ~~  {r.key}  (app_name={r.fact('app_name')!r} category={r.app_category!r})")
    # NO descriptor-domain flag (observability for the structured harvest): a project that yielded
    # content docs but no harvested `domain` facet. On the real corpus this is the small fraction of
    # projects without a descriptor — a reviewer can see which. (Facet name resolved generically.)
    no_cfg = [r for r in recs if r.fact("domain") is None and r.chunk_count > 0]
    lines.append(f"  NO descriptor domain (harvest gap): {len(no_cfg)}")
    for r in sorted(no_cfg, key=lambda x: x.key):
        lines.append(f"      cfg?  {r.key}")
    return "\n".join(lines)


def sidecar_dump() -> str:
    conn = connect()
    recs = all_projects(conn)
    conn.close()
    lines = ["METADATA SIDECAR (structured per-project records)"]
    if not recs:
        lines.append("  (empty — run `python -m rageval.ingest` first)")
    for r in recs:
        humor = "?" if r.has_humor is None else ("yes" if r.has_humor else "no")
        # Adapter FACTS printed generically (schema-agnostic): whatever the corpus declared/emitted,
        # with provenance. No app-specific field name is hardcoded here.
        facts_str = ", ".join(
            f"{k}={r.facts[k]!r}"
            + (f"[{r.facts_provenance.get(k)}]" if r.facts_provenance.get(k) else "")
            for k in sorted(r.facts)
        ) or "(none)"
        lines.append(
            f"  {r.key}\n"
            f"      publisher={r.publisher!r} category={r.app_category!r} "
            f"humor={humor} chunks={r.chunk_count}\n"
            f"      theme_tags={r.theme_tags} doc_types={r.doc_types}\n"
            f"      facts: {facts_str}\n"
            f"      summary={r.one_line_summary!r}"
        )
    # An example audit query, to teach the SQL-aggregation angle. publisher is the
    # TSV-authoritative facet; a NULL means no roster row (or no roster file).
    lines.append("")
    lines.append("  Example audit SQL:  SELECT publisher, COUNT(*) FROM projects "
                 "GROUP BY publisher;")
    return "\n".join(lines)


def quality_audit() -> str:
    """Heuristic content-quality flags over the stored chunks."""
    import re

    payloads = _hits_for(None)
    near_empty, codeish, dup = [], [], []
    seen: dict[str, str] = {}
    for pl in payloads:
        text = str(pl.get("text", ""))
        cid = str(pl.get("chunk_id", ""))
        stripped = text.strip()
        if len(stripped) < 40:
            near_empty.append(cid)
        # crude "looks like code/JSON not prose" heuristic: lots of braces/semicolons,
        # few sentence-ending periods relative to length.
        symbols = len(re.findall(r"[{};=<>]", text))
        if symbols > max(5, len(text) // 40):
            codeish.append(cid)
        norm = " ".join(stripped.lower().split())
        if norm in seen:
            dup.append((cid, seen[norm]))
        else:
            seen[norm] = cid
    lines = ["CONTENT-QUALITY AUDIT (heuristic flags)"]
    lines.append(f"  near-empty chunks : {len(near_empty)}")
    lines.append(f"  code/JSON-like    : {len(codeish)}  {codeish[:5]}")
    lines.append(f"  duplicate chunks  : {len(dup)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the ingested index + sidecar.")
    parser.add_argument("--project", help="Browse the chunks of one project (by project_id).")
    parser.add_argument("--coverage", action="store_true", help="Blind spots + chunk-count outliers.")
    parser.add_argument("--sidecar", action="store_true", help="Dump the structured metadata table.")
    parser.add_argument("--audit", action="store_true", help="Content-quality heuristic flags.")
    args = parser.parse_args()

    if args.project:
        print(browse_project(args.project))
    elif args.coverage:
        print(coverage_report())
    elif args.sidecar:
        print(sidecar_dump())
    elif args.audit:
        print(quality_audit())
    else:
        # Default: a quick overview combining sidecar + coverage.
        print(sidecar_dump())
        print()
        print(coverage_report())


if __name__ == "__main__":
    main()
