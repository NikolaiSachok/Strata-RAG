"""Metadata sidecar — a SQLite table of structured per-project records.

WHY a sidecar at all (the core design insight): target questions split into two classes,
and a vector top-k FAILS the second:

  * "which projects have a fruit-like theme?"        → SEMANTIC retrieval (embed → top-k)
  * "list every brand and count projects per brand"  → AGGREGATION (GROUP BY + COUNT)
  * "themes used in BOTH source-sets"                → SET INTERSECTION over a facet

A vector index cannot count or intersect — it returns the k nearest chunks, full stop.
So ingest produces BOTH a semantic index (Qdrant) AND this structured table, which
answers exact aggregation/filter/intersection queries with plain SQL.

It's also an OBSERVABILITY surface: `SELECT * FROM projects WHERE app_name IS NULL` finds
enrichment gaps; `GROUP BY publisher` aggregates the publisher facet; chunk-count
outliers flag bad include rules. SQLite is stdlib (no dependency) and its SQL is the clearest
way to teach the aggregation angle.

The sidecar file is gitignored — when built from a custom corpus it holds brand/theme data.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import SIDECAR_PATH


@dataclass
class ProjectRecord:
    """One structured row per project (produced by enrich.py).

    The last three fields are optional PROVENANCE/LINEAGE, derived from folder structure (not the
    LLM) by whichever adapter chooses to populate them:
      kotlin_source_id — for a project ported from a predecessor, the source project id (parsed
                         from a docx named "<new_id> (<source_id>)"). Enables the lineage query
                         "what was 2023 ported from?".
      status           — for non-conforming projects: released | unreleased | banned.
      non_conforming   — True for projects that don't follow the standard structure.
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
    # TWO DISTINCT "name" facets (deliberately split — they are NOT the same thing):
    #   app_name   — the project's PRODUCT / display name (a fictional e.g. "Frostline Fishing
    #                Log"). Source priority: config.yaml app.name > settings.md structured name >
    #                LLM-extracted product name. Stated in the docs, so the LLM/harvest extracts
    #                it confidently. (This CONSOLIDATES the old conflated product-name field.)
    #   publisher  — an AUTHORITATIVE label each project is associated with (a fictional e.g.
    #                "Maple Lagoon"), supplied via a roster TSV. It may DIFFER from the product
    #                title and be ABSENT from the docs, so it is NOT inferred by the LLM — it comes
    #                from a deterministic roster TSV join (roster.py), the authoritative ground
    #                truth. None when no roster row / no roster file.
    publisher: str | None = None
    # STRUCTURED METADATA columns filled by adapter-supplied StructuredFacts (#36) — the ADAPTER's
    # harvester lifts them from a per-project descriptor under a field-WHITELIST (never secrets);
    # the engine core knows no field names and just stores whatever facts an adapter emits into the
    # matching column. These power metadata queries the vector index can't serve (e.g. "website
    # URLs for fruit-themed apps?": theme from enrich + URL from here). The names below are the
    # generic structured SLOTS this sidecar offers; a corpus fills the ones its descriptor exposes.
    #   app_name       — see above (a descriptor name field is its highest-confidence source).
    #   domain         — the homepage/website host; landing_url is derived from it.
    #   landing_url    — DERIVED: "https://" + domain.
    #   app_number     — app.number (store/build number).
    #   bundle_id      — app.bundle_id.
    #   localization   — app.localization (e.g. "EN").
    #   contact_emails — best-effort, DERIVED from the public PHP templates (<local>@<domain>);
    #                    null when the template is too indirected to resolve honestly.
    #   contact_emails_derived — True iff contact_emails was constructed (provenance flag; the
    #                    address is not a verbatim literal, it's <localpart> joined to <domain>).
    app_name: str | None = None
    domain: str | None = None
    landing_url: str | None = None
    app_number: str | None = None
    bundle_id: str | None = None
    localization: str | None = None
    contact_emails: list[str] = field(default_factory=list)
    contact_emails_derived: bool | None = None
    # OBSERVABILITY: how confident we are in this project's enriched metadata.
    #   "high"  — a structured metadata source (settings.md) supplied the key fields.
    #   "low"   — only thin/absent sources were available (no brand/category extracted).
    #   None    — enrichment not run (no LLM backend); structural fields only.
    # The coverage report flags 'low' so a reviewer can see WHERE metadata is weak.
    metadata_confidence: str | None = None

    @property
    def key(self) -> str:
        return f"{self.source_set}/{self.project_id}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    key              TEXT PRIMARY KEY,
    project_id       TEXT NOT NULL,
    source_set       TEXT NOT NULL,
    app_category     TEXT,
    theme_tags       TEXT,   -- JSON array
    has_humor        INTEGER, -- 0/1/NULL
    doc_types        TEXT,   -- JSON array
    one_line_summary TEXT,
    chunk_count      INTEGER DEFAULT 0,
    kotlin_source_id TEXT,    -- port/predecessor lineage link (provenance)
    status           TEXT,    -- non-conforming status: released|unreleased|banned
    non_conforming   INTEGER, -- 1 if project does not follow the standard structure
    metadata_confidence TEXT, -- enrichment confidence: high|low|NULL(not run)
    publisher        TEXT,    -- authoritative publisher from the roster TSV join
    -- config.yaml harvest (deterministic, whitelist-only — NEVER secrets):
    app_name         TEXT,    -- app.name (the app's product/display name)
    domain           TEXT,    -- app.domain (homepage/website host)
    landing_url      TEXT,    -- DERIVED: https://<domain>
    app_number       TEXT,    -- app.number
    bundle_id        TEXT,    -- app.bundle_id
    localization     TEXT,    -- app.localization
    contact_emails   TEXT,    -- JSON array, best-effort DERIVED from PHP templates
    contact_emails_derived INTEGER  -- 1 if contact_emails was constructed (<local>@<domain>)
);
"""


# Columns added after the original schema shipped. `CREATE TABLE IF NOT EXISTS` won't add
# these to an existing DB, so we ALTER-ADD any that are missing — a tiny, safe migration so a
# schema bump never breaks an existing (gitignored) sidecar.
_MIGRATIONS = {
    "kotlin_source_id": "TEXT",
    "status": "TEXT",
    "non_conforming": "INTEGER",
    "metadata_confidence": "TEXT",
    "publisher": "TEXT",
    "app_name": "TEXT",
    "domain": "TEXT",
    "landing_url": "TEXT",
    "app_number": "TEXT",
    "bundle_id": "TEXT",
    "localization": "TEXT",
    "contact_emails": "TEXT",
    "contact_emails_derived": "INTEGER",
}


def connect(path: Path = SIDECAR_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    # Forward-migrate: add any columns missing from a pre-existing table.
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(projects)")}
    for col, sql_type in _MIGRATIONS.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE projects ADD COLUMN {col} {sql_type}")
    conn.commit()
    return conn


def upsert_project(conn: sqlite3.Connection, rec: ProjectRecord) -> None:
    conn.execute(
        """
        INSERT INTO projects (key, project_id, source_set, app_category,
                              theme_tags, has_humor, doc_types, one_line_summary, chunk_count,
                              kotlin_source_id, status, non_conforming, metadata_confidence,
                              publisher,
                              app_name, domain, landing_url, app_number, bundle_id, localization,
                              contact_emails, contact_emails_derived)
        VALUES (:key, :project_id, :source_set, :app_category,
                :theme_tags, :has_humor, :doc_types, :one_line_summary, :chunk_count,
                :kotlin_source_id, :status, :non_conforming, :metadata_confidence,
                :publisher,
                :app_name, :domain, :landing_url, :app_number, :bundle_id, :localization,
                :contact_emails, :contact_emails_derived)
        ON CONFLICT(key) DO UPDATE SET
            app_category=excluded.app_category,
            theme_tags=excluded.theme_tags, has_humor=excluded.has_humor,
            doc_types=excluded.doc_types, one_line_summary=excluded.one_line_summary,
            chunk_count=excluded.chunk_count, kotlin_source_id=excluded.kotlin_source_id,
            status=excluded.status, non_conforming=excluded.non_conforming,
            metadata_confidence=excluded.metadata_confidence,
            publisher=excluded.publisher,
            app_name=excluded.app_name, domain=excluded.domain, landing_url=excluded.landing_url,
            app_number=excluded.app_number, bundle_id=excluded.bundle_id,
            localization=excluded.localization, contact_emails=excluded.contact_emails,
            contact_emails_derived=excluded.contact_emails_derived
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
            "app_name": rec.app_name,
            "domain": rec.domain,
            "landing_url": rec.landing_url,
            "app_number": rec.app_number,
            "bundle_id": rec.bundle_id,
            "localization": rec.localization,
            "contact_emails": json.dumps(rec.contact_emails),
            "contact_emails_derived": (
                None if rec.contact_emails_derived is None else int(rec.contact_emails_derived)
            ),
        },
    )
    conn.commit()


def all_projects(conn: sqlite3.Connection) -> list[ProjectRecord]:
    rows = conn.execute("SELECT * FROM projects ORDER BY key").fetchall()
    out = []
    for r in rows:
        cols = r.keys()
        # GRACEFUL CONSOLIDATION: a pre-migration DB may still carry a legacy product-name column
        # ("brand", whose semantics were the product name). Fold it into app_name when app_name is
        # empty, so old sidecars keep their product-name signal under the new single field.
        app_name = r["app_name"]
        if not app_name and "brand" in cols:
            app_name = r["brand"]
        publisher = r["publisher"] if "publisher" in cols else None
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
                publisher=publisher,
                app_name=app_name,
                domain=r["domain"],
                landing_url=r["landing_url"],
                app_number=r["app_number"],
                bundle_id=r["bundle_id"],
                localization=r["localization"],
                contact_emails=json.loads(r["contact_emails"] or "[]"),
                contact_emails_derived=(
                    None if r["contact_emails_derived"] is None
                    else bool(r["contact_emails_derived"])
                ),
            )
        )
    return out


def to_dicts(records: list[ProjectRecord]) -> list[dict]:
    return [asdict(r) for r in records]
