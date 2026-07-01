"""Metadata sidecar — a SQLite structured store of per-project records.

WHY a sidecar at all (the core design insight): target questions split into two classes,
and a vector top-k FAILS the second:

  * "which projects have a fruit-like theme?"        → SEMANTIC retrieval (embed → top-k)
  * "list every publisher and count projects per one" → AGGREGATION (GROUP BY + COUNT)
  * "themes used in BOTH source-sets"                → SET INTERSECTION over a facet

A vector index cannot count or intersect — it returns the k nearest chunks, full stop.
So ingest produces BOTH a semantic index (Qdrant) AND this structured store, which
answers exact aggregation/filter/intersection queries with plain SQL.

SCHEMA-AGNOSTIC STRUCTURED STORE (#36 — the corpus-agnostic core):
  The store has TWO parts:

    1. `projects`      — GENERIC per-project columns the ENGINE owns: structural ids + the LLM
                         enrichment outputs (category/theme_tags/humor/summary) + provenance
                         (status/lineage) + the roster `publisher`. NO corpus-specific
                         structured-fact column lives here.
    2. `project_facts` — a GENERIC facet/EAV table: one row per (project, field) for every
                         adapter-emitted StructuredFact, with the value + its declared TYPE +
                         provenance. This is the SOURCE OF TRUTH for adapter facts. A corpus's
                         app-shaped fields (app_name/domain/bundle_id/…) are the SAMPLE ADAPTER's
                         DECLARED facets stored here — NOT hardcoded core columns. A brand-new
                         corpus emitting `premium`/`cause_of_loss`/… stores + queries them with NO
                         core edit.

  Aggregation (aggregate.py) queries a dynamic VIEW that pivots the declared facets onto columns
  and LEFT-JOINs `projects`, so count/list/group_by/lookup work over ANY declared facet exactly as
  they did over the old hardcoded columns — the queryable field set is the UNION of adapter-declared
  facets + the generic `projects` columns, validated against the DECLARATIONS, never a dataclass.

`ProjectRecord` remains a typed CONVENIENCE VIEW (generic columns as attributes + a `facts` dict for
the adapter facets); the generic facet store is the path that makes a non-app field queryable.

The sidecar file is gitignored — when built from a custom corpus it holds that corpus's data.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import SIDECAR_PATH
from .facts import coerce_facet_value


@dataclass
class ProjectRecord:
    """One structured per-project record (produced by enrich.py). A typed convenience view over the
    GENERIC `projects` columns PLUS a `facts` dict carrying whatever structured facts the adapter
    emitted (the schema-agnostic part). The engine core enumerates NO corpus facet name.

    Generic columns:
      project_id / source_set — identity (key = "source_set/project_id").
      app_category / theme_tags / has_humor / one_line_summary — the LLM ENRICHMENT outputs
                         (generic: any corpus gets a category/theme/summary; produced by the core
                         enricher, not a corpus descriptor).
      doc_types / chunk_count — structural bookkeeping.
      kotlin_source_id / status / non_conforming — optional PROVENANCE/LINEAGE from folder
                         structure (an adapter may populate them via folder_meta).
      publisher          — authoritative label from the roster TSV join (roster.py); never LLM.
      metadata_confidence— enrichment confidence (high|low|None).

    Adapter structured facts (the schema-agnostic store):
      facts              — {field: value} for every adapter-declared facet this project has (e.g.
                           {"app_name": "...", "domain": "..."} for the sample corpus, or
                           {"premium": 1200, "cause_of_loss": "..."} for another). Stored in
                           `project_facts`; queryable via the pivoted aggregation view.
      facts_provenance   — {field: provenance} companion (e.g. "descriptor" | "derived").
    """
    project_id: str
    source_set: str
    app_category: str | None = None
    theme_tags: list[str] = field(default_factory=list)
    has_humor: bool | None = None
    doc_types: list[str] = field(default_factory=list)
    one_line_summary: str | None = None
    chunk_count: int = 0
    kotlin_source_id: str | None = None
    status: str | None = None
    non_conforming: bool | None = None
    publisher: str | None = None
    metadata_confidence: str | None = None
    # The schema-agnostic adapter-fact store (per-project). Values are already coerced to their
    # declared facet type by the time they land here.
    facts: dict = field(default_factory=dict)
    facts_provenance: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source_set}/{self.project_id}"

    def fact(self, name: str, default=None):
        """Convenience accessor for one adapter fact (e.g. rec.fact('app_name'))."""
        return self.facts.get(name, default)


# --- GENERIC schema: the engine-owned `projects` columns + the facet/EAV table -----------------
# NO corpus-specific structured-fact column appears here. `projects` carries only generic
# structural + enrichment + provenance columns; adapter facts live in `project_facts`.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    key              TEXT PRIMARY KEY,
    project_id       TEXT NOT NULL,
    source_set       TEXT NOT NULL,
    app_category     TEXT,     -- LLM enrichment: a short product category
    theme_tags       TEXT,     -- JSON array (LLM enrichment: visual/narrative theme)
    has_humor        INTEGER,  -- 0/1/NULL (LLM enrichment)
    doc_types        TEXT,     -- JSON array (structural)
    one_line_summary TEXT,     -- LLM enrichment
    chunk_count      INTEGER DEFAULT 0,
    kotlin_source_id TEXT,     -- port/predecessor lineage link (provenance)
    status           TEXT,     -- non-conforming status: released|unreleased|banned
    non_conforming   INTEGER,  -- 1 if project does not follow the standard structure
    metadata_confidence TEXT,  -- enrichment confidence: high|low|NULL(not run)
    publisher        TEXT      -- authoritative publisher from the roster TSV join
);
CREATE TABLE IF NOT EXISTS project_facts (
    key         TEXT NOT NULL,   -- project key (source_set/project_id)
    project_id  TEXT NOT NULL,
    source_set  TEXT NOT NULL,
    field       TEXT NOT NULL,   -- the adapter-declared facet name
    value       TEXT,            -- the value, stored as text/JSON (typed per value_type)
    value_type  TEXT NOT NULL,   -- facet type: text|int|real|bool|text[]
    provenance  TEXT,            -- descriptor|derived|...
    PRIMARY KEY (key, field)
);
"""

# Migrations for a pre-existing DB. `CREATE TABLE IF NOT EXISTS` won't add missing columns; the
# facet table is new so it's created above. (The old hardcoded app columns, if present on a legacy
# DB, are simply ignored — the facet table + `facts` dict supersede them.)
_MIGRATIONS = {
    "kotlin_source_id": "TEXT",
    "status": "TEXT",
    "non_conforming": "INTEGER",
    "metadata_confidence": "TEXT",
    "publisher": "TEXT",
}


def _declared_facets() -> dict[str, str]:
    """{facet_name: facet_type} across all registered adapters (the queryable/writable allowlist).
    Local import so sidecar stays importable even if sources isn't fully wired."""
    from .sources.registry import all_declared_facets

    return all_declared_facets()


def _facts_view_sql(facets: dict[str, str]) -> str:
    """Build the pivoted VIEW that aggregate.py queries: `projects` LEFT-JOINed with one
    MAX(CASE ...) column per DECLARED facet, so every facet is a queryable column. Facet names come
    from adapter declarations (validated to a safe identifier) — never user/LLM input — so inlining
    them as column names is safe. A DB with no declared facets yields a view == `projects`."""
    cols = []
    for name in sorted(facets):
        if not name.replace("_", "").isalnum():  # defensive: declared names must be identifiers
            continue
        cols.append(
            f"MAX(CASE WHEN pf.field = '{name}' THEN pf.value END) AS \"{name}\"")
    facet_select = (",\n    " + ",\n    ".join(cols)) if cols else ""
    return f"""
CREATE TEMP VIEW projects_q AS
SELECT p.*{facet_select}
FROM projects p
LEFT JOIN project_facts pf ON pf.key = p.key
GROUP BY p.key;
"""


def connect(path: Path = SIDECAR_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    # Forward-migrate the generic `projects` table.
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(projects)")}
    for col, sql_type in _MIGRATIONS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE projects ADD COLUMN {col} {sql_type}")
    conn.commit()
    return conn


def facet_columns(facets: dict[str, str] | None = None) -> list[str]:
    """The declared facet names (queryable field allowlist), sorted. Defaults to all registered
    adapters' declared facets."""
    facets = _declared_facets() if facets is None else facets
    return sorted(n for n in facets if n.replace("_", "").isalnum())


def create_query_view(conn: sqlite3.Connection, facets: dict[str, str] | None = None) -> None:
    """(Re)create the TEMP pivoted view `projects_q` for THIS connection, one column per declared
    facet. aggregate.py calls this on its read-only connection before running a templated query."""
    facets = _declared_facets() if facets is None else facets
    conn.execute("DROP VIEW IF EXISTS projects_q")
    conn.executescript(_facts_view_sql(facets))


def upsert_project(conn: sqlite3.Connection, rec: ProjectRecord) -> None:
    """Write the generic `projects` row AND replace this project's `project_facts` rows.

    Fail-closed + type-safe: only facts whose name is a DECLARED facet are written, and each value
    is COERCED to its declared type. A mistyped value raises in coerce_facet_value — caught HERE so
    that ONE facet is skipped (never aborting the whole ingest batch); the skip is recorded via the
    returned/absent row (observable as a null facet)."""
    conn.execute(
        """
        INSERT INTO projects (key, project_id, source_set, app_category,
                              theme_tags, has_humor, doc_types, one_line_summary, chunk_count,
                              kotlin_source_id, status, non_conforming, metadata_confidence,
                              publisher)
        VALUES (:key, :project_id, :source_set, :app_category,
                :theme_tags, :has_humor, :doc_types, :one_line_summary, :chunk_count,
                :kotlin_source_id, :status, :non_conforming, :metadata_confidence,
                :publisher)
        ON CONFLICT(key) DO UPDATE SET
            app_category=excluded.app_category,
            theme_tags=excluded.theme_tags, has_humor=excluded.has_humor,
            doc_types=excluded.doc_types, one_line_summary=excluded.one_line_summary,
            chunk_count=excluded.chunk_count, kotlin_source_id=excluded.kotlin_source_id,
            status=excluded.status, non_conforming=excluded.non_conforming,
            metadata_confidence=excluded.metadata_confidence,
            publisher=excluded.publisher
        """,
        {
            "key": rec.key,
            "project_id": rec.project_id,
            "source_set": rec.source_set,
            "app_category": rec.app_category,
            "theme_tags": json.dumps(rec.theme_tags),
            "has_humor": None if rec.has_humor is None else int(rec.has_humor),
            "doc_types": json.dumps(rec.doc_types),
            "one_line_summary": rec.one_line_summary,
            "chunk_count": rec.chunk_count,
            "kotlin_source_id": rec.kotlin_source_id,
            "status": rec.status,
            "non_conforming": None if rec.non_conforming is None else int(rec.non_conforming),
            "metadata_confidence": rec.metadata_confidence,
            "publisher": rec.publisher,
        },
    )
    # Replace the project's facet rows (idempotent re-ingest).
    conn.execute("DELETE FROM project_facts WHERE key = ?", (rec.key,))
    facets = _declared_facets()
    for name, value in rec.facts.items():
        ftype = facets.get(name)
        if ftype is None:
            continue  # fail-closed: an undeclared facet is never stored
        try:
            coerced = coerce_facet_value(value, ftype)
        except (ValueError, TypeError):
            continue  # a MISTYPED fact degrades this ONE facet; the batch continues
        if coerced is None:
            continue
        stored = json.dumps(coerced) if ftype == "text[]" else str(coerced)
        conn.execute(
            "INSERT OR REPLACE INTO project_facts "
            "(key, project_id, source_set, field, value, value_type, provenance) "
            "VALUES (?,?,?,?,?,?,?)",
            (rec.key, rec.project_id, rec.source_set, name, stored, ftype,
             rec.facts_provenance.get(name)),
        )
    conn.commit()


def project_facts(conn: sqlite3.Connection, key: str) -> dict:
    """Read one project's facts → {field: value} (values decoded per their stored type)."""
    out: dict[str, object] = {}
    for r in conn.execute(
            "SELECT field, value, value_type FROM project_facts WHERE key = ?", (key,)):
        out[r["field"]] = _decode_fact(r["value"], r["value_type"])
    return out


def _decode_fact(value: str | None, value_type: str) -> object:
    if value is None:
        return None
    if value_type == "text[]":
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if value_type == "int":
        try:
            return int(value)
        except ValueError:
            return None
    if value_type == "real":
        try:
            return float(value)
        except ValueError:
            return None
    if value_type == "bool":
        return value in ("1", "True", "true")
    return value


def all_projects(conn: sqlite3.Connection) -> list[ProjectRecord]:
    rows = conn.execute("SELECT * FROM projects ORDER BY key").fetchall()
    out = []
    for r in rows:
        cols = r.keys()
        key = r["key"]
        facts = project_facts(conn, key)
        # Provenance per fact (for observability / the derived flag).
        prov = {
            fr["field"]: fr["provenance"]
            for fr in conn.execute(
                "SELECT field, provenance FROM project_facts WHERE key = ?", (key,))
        }
        out.append(
            ProjectRecord(
                project_id=r["project_id"],
                source_set=r["source_set"],
                app_category=r["app_category"],
                theme_tags=json.loads(r["theme_tags"] or "[]"),
                has_humor=None if r["has_humor"] is None else bool(r["has_humor"]),
                doc_types=json.loads(r["doc_types"] or "[]"),
                one_line_summary=r["one_line_summary"],
                chunk_count=r["chunk_count"],
                kotlin_source_id=r["kotlin_source_id"],
                status=r["status"],
                non_conforming=None if r["non_conforming"] is None else bool(r["non_conforming"]),
                metadata_confidence=r["metadata_confidence"],
                publisher=r["publisher"] if "publisher" in cols else None,
                facts=facts,
                facts_provenance=prov,
            )
        )
    return out


def to_dicts(records: list[ProjectRecord]) -> list[dict]:
    return [asdict(r) for r in records]
