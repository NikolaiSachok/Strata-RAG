"""Roster reconciliation report — reconcile the authoritative roster TSVs against the sidecar.

WHY this exists. The authoritative publisher for a project may NOT appear in its embedded text —
it lives in two out-of-band places that can DISAGREE:
  * the **roster TSV** — an authoritative `№ / ID / Publisher / Bundle` mapping (project folder
    id → publisher), but it can be STALE (a project gets re-assigned / re-titled over time).
  * the **sidecar** — the `app_name` (product title), populated either by `settings.md` (where
    present) or by LLM enrichment INFERRING a title from whatever survived in the docs.

This tool cross-checks them per project and classifies each as:
  * MATCH            — roster publisher == sidecar title (after normalisation).
  * MISMATCH         — they differ → a candidate for human adjudication (the project may have
                       been re-titled, or one source is stale).
  * sidecar-missing  — roster has a publisher, sidecar has no title → the title isn't stated in
                       the docs and the roster is the SOLE record (the expected case when docs
                       don't name it).
  * tsv-missing      — sidecar inferred a title the roster doesn't list.

IT DOES NOT AUTO-RESOLVE. Mismatches are FLAGGED, not decided. The authority order for a human
adjudicator is: `settings.md` title (explicit) > sidecar enrich-inferred > possibly-stale roster.

COVERAGE IS LIMITED BY DESIGN: when a project's docs don't name a title, that project will be
`sidecar-missing` — that's not an error, it's the expected shape (the roster is the record of
record). The MATCH/MISMATCH rows are the subset where a title survived in the docs.

Pure/deterministic over its inputs (roster rows + sidecar records) → unit-testable with synthetic
fixtures, no network.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# --- normalisation ----------------------------------------------------------

# Trailing caveats the data sometimes appends to a cell, e.g.
# "Acme (MUST NOT appear in the app)" or "Acme - do not show". We strip these so a value
# compares equal regardless of annotation. Conservative: only drop KNOWN caveat tails.
_CAVEAT_RE = re.compile(
    r"\s*(?:\(|-|—|;|,)?\s*(?:must not|do not|don'?t|cannot|can'?t|never)\b.*$",
    re.IGNORECASE,
)
_PARENS_RE = re.compile(r"\([^)]*\)")  # drop parenthetical notes entirely


def normalize_publisher(raw: str | None) -> str:
    """Normalise a publisher/title for comparison: drop parenthetical notes + trailing caveats,
    then case/space/punctuation-insensitive. Returns "" for an absent/blank value."""
    if not raw:
        return ""
    s = _PARENS_RE.sub(" ", raw)        # remove "(...)" notes
    s = _CAVEAT_RE.sub("", s)           # remove "... must not appear ..." tails
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)   # punctuation/space-insensitive
    return s.strip()


# --- TSV loading ------------------------------------------------------------

@dataclass(frozen=True)
class TsvRow:
    project_id: str   # the № column = the project folder id
    publisher: str    # raw publisher text (un-normalised; we normalise at compare time)
    source: str       # which TSV file it came from (for the report)


def load_tsv(path: Path) -> list[TsvRow]:
    """Parse a roster TSV (`№ / ID / Publisher / Bundle`, tab-separated, header row).

    We key on the № column (the project folder id). Tolerant of extra/blank columns and a
    missing publisher. The file STEM (e.g. 'northwind') is recorded as the source for the report."""
    rows: list[TsvRow] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(text.splitlines(), delimiter="\t")
    header = next(reader, None)  # skip the "№ ID Publisher Bundle" header
    for cols in reader:
        if not cols or not cols[0].strip():
            continue
        pid = cols[0].strip()
        publisher = cols[2].strip() if len(cols) > 2 else ""
        rows.append(TsvRow(project_id=pid, publisher=publisher, source=path.stem))
    return rows


# --- reconciliation ---------------------------------------------------------

@dataclass(frozen=True)
class RosterFinding:
    project_id: str
    tsv_source: str
    tsv_publisher: str    # raw
    sidecar_key: str      # "source_set/project_id" or "" if not in sidecar
    sidecar_title: str    # raw or ""
    status: str           # MATCH | MISMATCH | sidecar-missing | tsv-missing


# status order for stable sorting / counting
_STATUS_ORDER = ["MISMATCH", "MATCH", "sidecar-missing", "tsv-missing"]


def reconcile(tsv_rows: list[TsvRow],
              sidecar: list,  # list[ProjectRecord]-like with .project_id/.source_set/.app_name
              ) -> list[RosterFinding]:
    """Cross-check the authoritative roster publisher against the sidecar's doc-inferred app_name,
    keyed by project_id.

    The sidecar's `publisher` IS the roster value (a join), so comparing those would be
    tautological. The interesting reconciliation is the roster's authoritative publisher vs the
    SURFACE app_name the docs/LLM inferred — a divergence flags where the product title and the
    authoritative publisher differ (→ sidecar-missing/MISMATCH for human review).

    A project_id can appear under several sidecar source_sets (e.g. northwind vs northwind-extra);
    we match a roster row to ANY sidecar record sharing its project_id. Sidecar records with a
    project_id absent from every roster become `tsv-missing`."""
    # Index the sidecar by project_id → list of (key, app_name). app_name is a corpus FACET
    # (schema-agnostic store); a record exposes it via .fact('app_name') (None for a corpus that
    # doesn't declare it, handled as "" — roster is then the sole record).
    def _app_name(rec) -> str:
        getter = getattr(rec, "fact", None)
        return (getter("app_name") if callable(getter) else getattr(rec, "app_name", None)) or ""

    side_by_pid: dict[str, list[tuple[str, str]]] = {}
    for rec in sidecar:
        side_by_pid.setdefault(rec.project_id, []).append(
            (f"{rec.source_set}/{rec.project_id}", _app_name(rec)))

    findings: list[RosterFinding] = []
    matched_pids: set[str] = set()

    for row in tsv_rows:
        matches = side_by_pid.get(row.project_id)
        if not matches:
            # In the roster but never ingested into the sidecar (not in the corpus we ingested).
            findings.append(RosterFinding(row.project_id, row.source, row.publisher, "", "",
                                          "sidecar-missing"))
            continue
        matched_pids.add(row.project_id)
        for key, side_title in matches:
            if not side_title:
                status = "sidecar-missing"  # title not stated in docs → roster is sole record
            elif normalize_publisher(row.publisher) == normalize_publisher(side_title):
                status = "MATCH"
            else:
                status = "MISMATCH"          # candidate — human adjudication
            findings.append(RosterFinding(row.project_id, row.source, row.publisher, key,
                                          side_title, status))

    # Sidecar projects whose project_id is in NO roster → tsv-missing (sidecar inferred a title).
    tsv_pids = {r.project_id for r in tsv_rows}
    for rec in sidecar:
        if rec.project_id not in tsv_pids and _app_name(rec):
            findings.append(RosterFinding(rec.project_id, "", "",
                                          f"{rec.source_set}/{rec.project_id}", _app_name(rec),
                                          "tsv-missing"))
    return findings


def counts(findings: list[RosterFinding]) -> Counter:
    return Counter(f.status for f in findings)


def render_report(findings: list[RosterFinding]) -> str:
    """A table grouped by status (mismatches first — those need a human), plus counts."""
    c = counts(findings)
    lines = ["ROSTER RECONCILIATION REPORT (roster vs sidecar — candidates flagged, NOT auto-resolved)",
             "=" * 78]
    order = {s: i for i, s in enumerate(_STATUS_ORDER)}
    for f in sorted(findings, key=lambda x: (order.get(x.status, 9), x.project_id)):
        key = f.sidecar_key or "—"
        lines.append(f"  [{f.status:<16}] {f.project_id:<10} "
                     f"roster={f.tsv_publisher or '—'!r:<24} sidecar={f.sidecar_title or '—'!r:<24} {key}")
    lines.append("-" * 78)
    lines.append("counts: " + ", ".join(f"{s}={c[s]}" for s in _STATUS_ORDER if c[s]))
    lines.append(
        "\nNote: when a project's title isn't stated in its docs, 'sidecar-missing' is the "
        "EXPECTED outcome (the roster is the record of record). MISMATCH rows are candidates "
        "for human review; authority = settings.md title > sidecar enrich > possibly-stale roster.")
    return "\n".join(lines)


def run(tsv_paths: list[Path], sidecar_records: list) -> list[RosterFinding]:
    """Top-level: load every roster TSV, reconcile against the given sidecar records."""
    rows: list[TsvRow] = []
    for p in tsv_paths:
        rows.extend(load_tsv(p))
    return reconcile(rows, sidecar_records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Roster reconciliation: reconcile authoritative roster TSV(s) against the "
                    "sidecar. Coverage is LIMITED when a project's title isn't stated in its docs "
                    "— expect mostly 'sidecar-missing'; the few MATCH/MISMATCH rows are the signal.")
    parser.add_argument("--tsv", nargs="+", required=True, type=Path,
                        help="One or more roster TSVs (cols: № ID Publisher Bundle). Paths from "
                             "args; never hardcoded (custom-corpus rosters are gitignored).")
    parser.add_argument("--sidecar", type=Path, default=None,
                        help="Sidecar sqlite path (default: the configured rageval.sqlite).")
    args = parser.parse_args()

    from .config import SIDECAR_PATH
    from .sidecar import all_projects, connect

    conn = connect(args.sidecar or SIDECAR_PATH)
    records = all_projects(conn)
    conn.close()

    findings = run(args.tsv, records)
    print(render_report(findings))


if __name__ == "__main__":
    main()
