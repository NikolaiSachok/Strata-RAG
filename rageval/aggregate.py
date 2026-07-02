"""Aggregate: the TEMPLATED-INTENT executor over the metadata sidecar.

This is the second of the engine's two query classes (see sidecar.py): a vector top-k
cannot COUNT, GROUP BY, or fetch an exact row, so structural/aggregation questions are
answered here with plain SQL against the SQLite sidecar instead of the embedding index.

THE DESIGN — "LLM proposes, deterministic rules enforce" (the trust boundary):
    The LLM is good at turning a natural-language question into a STRUCTURED INTENT
    ({intent, field, filter, aggregate}). It is NOT trusted to write SQL. Each intent
    fills exactly ONE vetted, parameterized query TEMPLATE; every slot value is validated
    against the REAL sidecar schema (a column whitelist DERIVED from ProjectRecord) before
    anything runs. Unknown field / unknown intent / malformed filter → REJECT (never
    execute). This is text-to-intent, not text-to-SQL — a strictly smaller, safer surface.
    (The richer free-form text-to-SQL with a generated-query validator is issue #5.)

WHY this is safe even though an LLM chose the parameters:
    * The query STRINGS are hard-coded templates in THIS file — the LLM never supplies SQL.
    * Field/column names are checked against a whitelist; a value that isn't a real column
      can't reach the query at all.
    * Filter VALUES are bound as SQL parameters (never string-interpolated) → no injection.
    * The connection is opened READ-ONLY and every template carries a mandatory LIMIT.

The result carries the executed query + bound params so the caller can show EXACTLY what
ran (the transparency block on the API response).
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .config import SIDECAR_PATH
from .facts import RESERVED_FACET_NAMES
from .sidecar import create_query_view

# ---------------------------------------------------------------------------
# The trust boundary: what the LLM is allowed to name.
# ---------------------------------------------------------------------------

# The GENERIC engine-owned columns on `projects` (structural + enrichment + provenance), plus the
# synthesized primary `key`. NO corpus-specific structured-fact name lives here. SINGLE SOURCE OF
# TRUTH is facts.RESERVED_FACET_NAMES — the same set FacetSpec forbids a facet from colliding with,
# so "generic column" means exactly one thing across the engine.
_GENERIC_FIELDS: frozenset[str] = RESERVED_FACET_NAMES

# Generic JSON-array columns (stored as TEXT holding a JSON list). Equality filters and DISTINCT
# over these are imperfect (they match the serialized blob), so we only allow them where it's
# honest. Adapter facets of declared type text[] are added dynamically in _json_array_fields().
_GENERIC_JSON_ARRAY_FIELDS: frozenset[str] = frozenset({"theme_tags", "doc_types"})


def _declared_facets() -> dict[str, str]:
    """{facet_name: facet_type} across all registered adapters (queryable allowlist). Resolved at
    CALL time so a corpus registered after import (a plugin) is honoured."""
    from .sources.registry import all_declared_facets

    return all_declared_facets()


def allowed_fields() -> frozenset[str]:
    """The QUERYABLE field allowlist: the generic `projects` columns UNIONed with every
    adapter-DECLARED facet name (#36). Derived from adapter declarations — NOT a dataclass — so a
    corpus's own facets are queryable with no core edit, and an undeclared field is rejected
    (fail-closed)."""
    return _GENERIC_FIELDS | frozenset(_declared_facets())


def _json_array_fields() -> frozenset[str]:
    """Generic JSON-array columns + any declared facet of type text[]."""
    return _GENERIC_JSON_ARRAY_FIELDS | frozenset(
        n for n, t in _declared_facets().items() if t == "text[]")


# Back-compat module constant (a SNAPSHOT at import). The executor validates against the live
# allowed_fields() so a plugin-registered facet is accepted even if it registered after import.
ALLOWED_FIELDS: frozenset[str] = allowed_fields()

# The intents this executor supports. Each maps to ONE parameterized template below.
ALLOWED_INTENTS: frozenset[str] = frozenset(
    {"count", "list", "group_by_count", "top_n", "lookup"}
)

# A conservative hard cap so a templated query can never scan/emit unbounded rows.
MAX_LIMIT = 1000
DEFAULT_LIMIT = 100


class AggregateError(RuntimeError):
    """Raised when a slot fails validation (unknown field/intent, bad filter). The caller
    treats this as a signal to FALL BACK to the semantic path, not as a fatal error."""


# C0 (U+0000–U+001F except tab/newline) + DEL + C1 (U+007F–U+009F) control characters. Fact VALUES
# now include spreadsheet cells (untrusted); a raw control byte (ESC `\x1b[…`, BEL) rendered into the
# answer could corrupt a terminal/log. Strip them so the deterministic render is safe to display.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _sanitize_answer(text: str) -> str:
    """Strip control chars from a rendered aggregation answer (the values may be untrusted cells)."""
    return _CONTROL_CHARS_RE.sub("", text)


def format_aggregation_answer(result: "AggregateResult") -> str:
    """Render a templated-query result into a short human answer string. Kept simple and
    deterministic (no LLM): the structured rows + the routing block carry the real detail.

    Public (was dispatch._format_aggregation_answer): both dispatch.py and the agent compose
    aggregation observations, so it lives here next to AggregateResult — its natural home.

    SECURITY: fact values now include untrusted spreadsheet cells, so the rendered string is
    control-char-sanitized here (consistently, so dispatch's #16 invariant — answer == renderer
    output — still holds). Injection/exfil SCANNING of the values happens in dispatch/agent, which
    populate the GuardrailReport; this function only makes the string safe to serialize."""
    rows = result.rows
    if result.intent == "count":
        n = rows[0].get("count") if rows else 0
        return _sanitize_answer(f"Count: {n}.")
    if result.intent in ("group_by_count", "top_n"):
        parts = [f"{r.get(k)}: {r.get('count')}" for r in rows for k in r if k != "count"]
        body = "; ".join(parts) if parts else "no rows"
        label = "Top results" if result.intent == "top_n" else "Grouped counts"
        return _sanitize_answer(f"{label} — {body}.")
    if result.intent == "list":
        # The single non-derived column holds the distinct values.
        vals = [str(next(iter(r.values()))) for r in rows]
        return _sanitize_answer("Distinct values: " + (", ".join(vals) if vals else "(none)."))
    if result.intent == "lookup":
        if not rows:
            return "No matching record."
        return _sanitize_answer("Record: " + "; ".join(f"{k}={v}" for k, v in rows[0].items()))
    return f"{len(rows)} row(s)."  # pragma: no cover


@dataclass
class AggregateResult:
    """The outcome of one templated query — enough to render AND to audit (transparency)."""
    intent: str
    rows: list[dict]
    executed_query: str            # the exact SQL template that ran
    params: list = field(default_factory=list)  # the bound parameters (no interpolation)
    row_count: int = 0

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "rows": self.rows,
            "executed_query": self.executed_query,
            "params": list(self.params),
            "row_count": self.row_count,
        }


# ---------------------------------------------------------------------------
# Slot validation (enforce, don't trust).
# ---------------------------------------------------------------------------

def _validate_field(name: str | None, *, what: str) -> str:
    """Confirm a slot names a queryable field: a generic `projects` column OR an adapter-DECLARED
    facet. Rejects anything off the live allowlist (resolved at call time so a plugin-registered
    facet is accepted).

    This is the core of the trust boundary: an LLM-proposed field that isn't a real column/facet
    can NEVER reach a query — it's refused here first."""
    allowed = allowed_fields()
    if not name or name not in allowed:
        raise AggregateError(
            f"{what} '{name}' is not a known sidecar field "
            f"(allowed: {', '.join(sorted(allowed))})."
        )
    return name


def _normalize_filter(filt: dict | None) -> dict[str, object]:
    """Validate a {field: value} filter map: every KEY must be a whitelisted column. VALUES
    are returned untouched (they are bound as parameters, never interpolated). A None/empty
    filter is a valid 'no filter'."""
    if not filt:
        return {}
    if not isinstance(filt, dict):
        raise AggregateError(f"filter must be an object of field:value, got {type(filt).__name__}.")
    clean: dict[str, object] = {}
    for fname, value in filt.items():
        _validate_field(fname, what="filter field")
        clean[fname] = value
    return clean


def _clamp_limit(limit: int | None) -> int:
    if not limit or limit <= 0:
        return DEFAULT_LIMIT
    return min(int(limit), MAX_LIMIT)


def _qi(name: str) -> str:
    """Quote a validated column/facet identifier for interpolation into a template.

    The name is ALREADY whitelisted (a generic column or a declared facet, both validated to a
    strict SQL identifier at declaration) — never user/LLM input — so this is not an
    injection guard but a CORRECTNESS one: double-quoting lets a facet whose name happens to be a
    SQL keyword (e.g. `order`, `group`) still be queried, matching how the pivot view quotes its
    facet columns. Belt-and-braces: reject an embedded double-quote (impossible past validation)."""
    if '"' in name:  # pragma: no cover - validation already forbids this
        raise AggregateError(f"invalid identifier {name!r}")
    return f'"{name}"'


def _where_clause(filt: dict[str, object]) -> tuple[str, list]:
    """Build a parameterized WHERE from a validated filter map. Keys are already whitelisted
    column names (quoted for safe interpolation); VALUES are bound as ? params. Booleans map to
    0/1 to match the sidecar's INTEGER storage; None means IS NULL."""
    if not filt:
        return "", []
    parts: list[str] = []
    params: list = []
    for fname, value in filt.items():
        col = _qi(fname)
        if value is None:
            parts.append(f"{col} IS NULL")
        elif isinstance(value, bool):
            parts.append(f"{col} = ?")
            params.append(1 if value else 0)
        else:
            parts.append(f"{col} = ?")
            params.append(value)
    return " WHERE " + " AND ".join(parts), params


# ---------------------------------------------------------------------------
# Read-only connection.
# ---------------------------------------------------------------------------

def _connect_readonly(path: Path) -> sqlite3.Connection:
    """Open the sidecar READ-ONLY. The aggregation path only ever reads; opening with
    mode=ro means even a buggy/hostile template literally CANNOT write (sqlite enforces it
    at the driver level — defense in depth behind the templates being SELECT-only)."""
    # file: URI with mode=ro. immutable=0 so a concurrently-written DB is still read correctly.
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# The vetted templates — one per intent.
# ---------------------------------------------------------------------------

def execute(
    intent: str,
    *,
    field: str | None = None,
    filter: dict | None = None,
    limit: int | None = None,
    sidecar_path: Path | None = None,
) -> AggregateResult:
    """Validate the slots, fill the matching template, and run it READ-ONLY.

    Slots:
      intent  — one of ALLOWED_INTENTS.
      field   — the column to list / group by / look up by (required for some intents).
      filter  — {column: value} equality filter (each column whitelisted; values bound).
      limit   — row cap (clamped to MAX_LIMIT; templates always carry a LIMIT).

    Raises AggregateError on any validation failure (caller falls back to semantic).
    """
    if intent not in ALLOWED_INTENTS:
        raise AggregateError(
            f"intent '{intent}' is not supported "
            f"(allowed: {', '.join(sorted(ALLOWED_INTENTS))})."
        )
    # Resolve the sidecar path at CALL time (not as a default arg) so a test/caller that
    # monkeypatches the module-level SIDECAR_PATH is honoured.
    if sidecar_path is None:
        sidecar_path = SIDECAR_PATH
    filt = _normalize_filter(filter)
    lim = _clamp_limit(limit)

    if intent == "count":
        where, params = _where_clause(filt)
        sql = f"SELECT COUNT(*) AS count FROM projects_q{where} LIMIT ?"
        params = params + [lim]
        rows = _run(sidecar_path, sql, params)
        return AggregateResult("count", rows, sql, params, row_count=len(rows))

    if intent == "list":
        col = _validate_field(field, what="list field")
        qcol = _qi(col)
        where, params = _where_clause(filt)
        # DISTINCT values of one column (the common "list all brands / categories" question).
        sql = f"SELECT DISTINCT {qcol} AS {qcol} FROM projects_q{where} ORDER BY {qcol} LIMIT ?"
        params = params + [lim]
        rows = _run(sidecar_path, sql, params)
        return AggregateResult("list", rows, sql, params, row_count=len(rows))

    if intent == "group_by_count":
        col = _validate_field(field, what="group-by field")
        qcol = _qi(col)
        where, params = _where_clause(filt)
        sql = (
            f"SELECT {qcol} AS {qcol}, COUNT(*) AS count FROM projects_q{where} "
            f"GROUP BY {qcol} ORDER BY count DESC, {qcol} LIMIT ?"
        )
        params = params + [lim]
        rows = _run(sidecar_path, sql, params)
        return AggregateResult("group_by_count", rows, sql, params, row_count=len(rows))

    if intent == "top_n":
        # top-N is group_by_count with a tighter LIMIT (the N). Same vetted template.
        col = _validate_field(field, what="top-n field")
        qcol = _qi(col)
        where, params = _where_clause(filt)
        sql = (
            f"SELECT {qcol} AS {qcol}, COUNT(*) AS count FROM projects_q{where} "
            f"GROUP BY {qcol} ORDER BY count DESC, {qcol} LIMIT ?"
        )
        params = params + [lim]
        rows = _run(sidecar_path, sql, params)
        return AggregateResult("top_n", rows, sql, params, row_count=len(rows))

    if intent == "lookup":
        # Degenerate aggregation: fetch a single row's fields by an id filter. The FILTER
        # carries the id ({project_id: "2023"} or {key: "northwind/2023"}); `field` optionally
        # restricts which column to return (else the whole row).
        if not filt:
            raise AggregateError("lookup requires a filter identifying the row (e.g. {key: ...}).")
        where, params = _where_clause(filt)
        if field is not None:
            col = _validate_field(field, what="lookup field")
            qcol = _qi(col)
            sql = f"SELECT key, {qcol} AS {qcol} FROM projects_q{where} LIMIT ?"
        else:
            sql = f"SELECT * FROM projects_q{where} LIMIT ?"
        params = params + [lim]
        rows = _run(sidecar_path, sql, params)
        return AggregateResult("lookup", rows, sql, params, row_count=len(rows))

    # Unreachable (intent validated above), but keep the boundary explicit.
    raise AggregateError(f"intent '{intent}' has no template.")  # pragma: no cover


def _run(sidecar_path: Path, sql: str, params: list) -> list[dict]:
    """Execute a vetted, parameterized SELECT on a READ-ONLY connection.

    Belt-and-braces: assert the statement is a single SELECT before running it, so even a
    future edit can't smuggle a write through this path."""
    stmt = sql.strip().rstrip(";")
    if ";" in stmt or not re.match(r"(?is)^select\b", stmt):
        raise AggregateError("internal: only single SELECT statements may execute here.")
    try:
        # Self-hardened: a connect-time failure (missing file, perms, locked/corrupt DB) raises a
        # raw sqlite3.Error/OSError BEFORE any query runs. Catch it here too so this layer degrades
        # to AggregateError on its own — belt-and-braces behind the agent's own catch.
        conn = _connect_readonly(Path(sidecar_path))
    except (sqlite3.Error, OSError) as e:
        raise AggregateError(f"sidecar unavailable: {e}") from e
    try:
        # Build the pivoted TEMP view (`projects_q`) the templates query — one column per declared
        # facet, LEFT-JOINed onto `projects`. A TEMP view is allowed on a mode=ro main DB (sqlite's
        # temp store is writable), so this keeps the main sidecar strictly read-only while making
        # every adapter facet a queryable column.
        create_query_view(conn)
        cur = conn.execute(stmt, params)
        return _rows_to_dicts(cur.fetchall())
    except sqlite3.OperationalError as e:
        # e.g. the sidecar file doesn't exist yet / no such table → fall back to semantic.
        raise AggregateError(f"sidecar query failed: {e}") from e
    finally:
        conn.close()


def parse_slots(raw: str) -> dict:
    """Parse the LLM's structured slot output (JSON) → {intent, field, filter, limit}.

    Tolerant of code fences / stray prose around the object (same contract as eval/enrich).
    Returns a plain dict; validation happens in execute()."""
    raw = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start, end = raw.find("{"), raw.rfind("}")
        candidate = raw[start : end + 1] if start != -1 and end > start else "{}"
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
