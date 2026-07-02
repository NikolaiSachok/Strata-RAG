"""Ingestion observability — the dry-run manifest + coverage report.

This is a FIRST-CLASS deliverable, not an afterthought. Embedding is the LAST step of
ingest; the pipeline must be inspectable BEFORE you spend money/time embedding. A manifest
answers, without embedding anything:

  * INCLUDED — what WOULD be ingested, by type, with a chunk/token/cost estimate.
  * EXCLUDED — every dropped file AND the exact rule that dropped it (catches both excess
    junk AND real docs wrongly filtered out).
  * COVERAGE — projects with >=1 content doc, projects with ZERO (blind spots), and
    per-project doc-count OUTLIERS (e.g. 10x the median → likely a bad include rule).

The loop this enables — inspect -> spot excess/gaps -> refine corpus-rules.yaml -> re-run
-> diff — is the whole point: a fast, repeatable way to get discovery right before paying
to embed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .chunking import chunk_text
from .classify import CorpusRules, PolicyResolver, classify
from .config import SETTINGS, Settings
from .redact import redact
from .sources.base import SourceDoc


@dataclass
class IncludedEntry:
    doc: SourceDoc
    n_chunks: int
    n_chars: int
    n_redactions: int = 0  # secret values that WOULD be scrubbed from this doc at ingest
    n_pii: int = 0         # PII (emails) that WOULD be scrubbed from this doc at ingest
    metadata_only: bool = False  # routed to enrich, NOT embedded (e.g. settings.md)


@dataclass
class ExcludedEntry:
    doc: SourceDoc
    reason: str


@dataclass
class Coverage:
    projects_with_content: list[str] = field(default_factory=list)   # "source_set/project_id"
    blind_spots: list[str] = field(default_factory=list)             # zero included docs
    outliers: list[tuple[str, int]] = field(default_factory=list)    # (project, doc_count)


@dataclass
class Manifest:
    included: list[IncludedEntry]
    excluded: list[ExcludedEntry]
    coverage: Coverage
    total_chunks: int
    total_chars: int
    total_redactions: int = 0  # secret values that WOULD be scrubbed across all included docs
    total_pii: int = 0         # PII (emails) that WOULD be scrubbed across all included docs

    # A rough cost estimate. Local embeddings are free; we report tokens so the number is
    # meaningful if you swap to a paid embedder. ~4 chars/token is the usual rule of thumb.
    @property
    def est_tokens(self) -> int:
        return self.total_chars // 4


def _all_project_ids(docs: list[SourceDoc]) -> set[str]:
    return {f"{d.source_set}/{d.project_id}" for d in docs}


def build_manifest(all_docs: list[SourceDoc], rules: CorpusRules,
                   settings: Settings = SETTINGS) -> Manifest:
    """Classify every discovered doc and compute the include/exclude + coverage plan.

    Pure over (docs, rules, settings) → deterministic and unit-testable on a fixture.
    Does NOT embed and does NOT touch Qdrant.
    """
    included: list[IncludedEntry] = []
    excluded: list[ExcludedEntry] = []
    total_chunks = 0
    total_chars = 0
    total_redactions = 0
    total_pii = 0

    # project -> number of INCLUDED docs (for coverage + outlier detection)
    included_per_project: dict[str, int] = {}

    # Resolve each doc's adapter ClassificationPolicy (#37), cached by source_set.
    resolver = PolicyResolver()

    for doc in all_docs:
        dec = classify(doc, rules, resolver.policy_for(doc.source_set))
        if dec.include:
            # Plan the SAME redaction the ingest will do (secrets AND PII), so the manifest's
            # chunk/char estimate matches reality and BOTH guardrail counts are visible before
            # embedding.
            clean_text, n_red, n_pii = redact(doc.raw_text, doc_type=doc.doc_type,
                                               pii_policy=rules.pii_policy)
            n_chars = len(clean_text)
            # METADATA-ONLY docs (settings.md) are NOT embedded — enrich consumes them — so they
            # contribute ZERO chunks to the index. We still count their redactions (scrubbed at
            # ingest before enrich sees them) and mark them so the manifest shows them as
            # enrich-only rather than mis-reporting embedded chunks.
            if dec.metadata_only:
                pieces = []
            else:
                pieces = chunk_text(clean_text, size=settings.chunk_size,
                                    overlap=settings.chunk_overlap)
            included.append(IncludedEntry(doc=doc, n_chunks=len(pieces), n_chars=n_chars,
                                          n_redactions=n_red, n_pii=n_pii,
                                          metadata_only=dec.metadata_only))
            total_chunks += len(pieces)
            total_chars += n_chars
            total_redactions += n_red
            total_pii += n_pii
            key = f"{doc.source_set}/{doc.project_id}"
            included_per_project[key] = included_per_project.get(key, 0) + 1
        else:
            excluded.append(ExcludedEntry(doc=doc, reason=dec.reason))

    # COVERAGE: every project the adapters SAW (even if all its docs were excluded) must
    # be considered, so a project that is pure noise shows up as a blind spot.
    seen_projects = _all_project_ids(all_docs)
    with_content = sorted(k for k, n in included_per_project.items() if n > 0)
    blind = sorted(seen_projects - set(with_content))

    # OUTLIERS: projects whose included-doc count is far above the median (>= 3x and
    # above the median by at least 2) — a likely sign a bad include rule swept in junk.
    counts = list(included_per_project.values())
    outliers: list[tuple[str, int]] = []
    if len(counts) >= 3:
        med = statistics.median(counts)
        for k, n in sorted(included_per_project.items()):
            if med > 0 and n >= max(3 * med, med + 2):
                outliers.append((k, n))

    return Manifest(
        included=included,
        excluded=excluded,
        coverage=Coverage(projects_with_content=with_content, blind_spots=blind,
                          outliers=outliers),
        total_chunks=total_chunks,
        total_chars=total_chars,
        total_redactions=total_redactions,
        total_pii=total_pii,
    )


def render_manifest(m: Manifest, settings: Settings = SETTINGS) -> str:
    """Human-readable manifest for `ingest --dry-run`."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("INGEST DRY-RUN MANIFEST (no embedding performed)")
    lines.append("=" * 72)
    lines.append(f"corpus_root : {settings.corpus_root}")
    lines.append(f"corpus_intent: {settings.corpus_intent}")
    lines.append("")

    # INCLUDED, grouped by doc_type for a quick "by type" view.
    by_type: dict[str, list[IncludedEntry]] = {}
    for e in m.included:
        by_type.setdefault(e.doc.doc_type, []).append(e)
    lines.append(f"INCLUDED — {len(m.included)} docs, "
                 f"{m.total_chunks} chunks, ~{m.est_tokens} tokens (est.); "
                 f"secrets to redact: {m.total_redactions}; PII (emails) to redact: {m.total_pii}")
    for dt in sorted(by_type):
        entries = by_type[dt]
        chunks = sum(e.n_chunks for e in entries)
        lines.append(f"  [{dt}] {len(entries)} docs, {chunks} chunks")
        for e in sorted(entries, key=lambda x: x.doc.doc_id):
            # Flag any doc that carries secrets / PII so a reviewer sees WHERE redaction fires.
            flags = []
            if e.metadata_only:
                flags.append("enrich-only, NOT embedded")
            if e.n_redactions:
                flags.append(f"redacts {e.n_redactions} secret(s)")
            if e.n_pii:
                flags.append(f"redacts {e.n_pii} PII email(s)")
            red = f"  [{'; '.join(flags)}]" if flags else ""
            lines.append(f"      + {e.doc.doc_id}  ({e.n_chunks} chunks){red}")
    lines.append("")

    # EXCLUDED, with the rule that dropped each.
    lines.append(f"EXCLUDED — {len(m.excluded)} docs (each with the rule that dropped it)")
    for e in sorted(m.excluded, key=lambda x: x.doc.doc_id):
        lines.append(f"      - {e.doc.doc_id}  [{e.reason}]")
    lines.append("")

    # COVERAGE.
    cov = m.coverage
    lines.append("COVERAGE")
    lines.append(f"  projects with content : {len(cov.projects_with_content)}")
    for p in cov.projects_with_content:
        lines.append(f"      ok  {p}")
    lines.append(f"  BLIND SPOTS (zero content docs) : {len(cov.blind_spots)}")
    for p in cov.blind_spots:
        lines.append(f"      !!  {p}")
    lines.append(f"  doc-count outliers : {len(cov.outliers)}")
    for p, n in cov.outliers:
        lines.append(f"      ??  {p}  ({n} docs)")
    lines.append("=" * 72)
    return "\n".join(lines)
