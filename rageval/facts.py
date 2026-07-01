"""Structured facts — the corpus-agnostic contract for adapter-supplied sidecar metadata.

WHY this module exists (the Phase-4 decouple, issue #36):
  Some corpora ship the most AUTHORITATIVE structured facts about a project OUTSIDE its
  narrative documents — in a per-project descriptor file (a `config.yaml`, a manifest, a
  spreadsheet). Those facts belong in the metadata SIDECAR (structured → SQL facet), not the
  vector index. But WHICH fields a descriptor exposes, and HOW they map to sidecar columns, is
  entirely corpus-specific. If the engine core hard-coded one corpus's field names it would be
  single-corpus by construction.

  So the core defines only the SHAPE of a structured fact and a REUSABLE, security-hardened
  harvesting PRIMITIVE. An adapter (behind the `SourceAdapter.harvest_facts` hook) decides which
  fields exist and emits `StructuredFact`s; the core consumes whatever it emits and knows NO
  field names.

SECURITY — the whitelist / fail-closed invariant lives HERE, generically:
  Descriptor files routinely interleave clean fields with SECRET blocks (api keys, tokens,
  session ids). The primitive `FieldWhitelistHarvester` lifts ONLY the keys an adapter opts in
  (a WHITELIST, never a blacklist) and, belt-and-braces, refuses any value whose KEY looks
  secret — so a whitelist can't be defeated by a renamed field, and a NEW secret field added
  upstream is excluded BY DEFAULT (fail-closed, the only safe default for credential-adjacent
  data). This mechanism is corpus-neutral: it enforces the invariant without knowing a single
  concrete field name (the adapter supplies the whitelist).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Belt-and-braces guard: even a WHITELISTED key is refused if its NAME looks like a secret.
# Corpus-neutral (matches on the shape of credential-adjacent key names), so it lives in core.
# Kept CONSISTENT with redact._SECRET_KEY_WORDS (the ingest-time secret notion) so the two agree
# on what "looks secret" — this catches standalone password/credential/login/apikey too, not only
# the underscore-prefixed forms. (This is belt-and-braces; the whitelist is the primary defense.)
_SECRET_KEY_RE = re.compile(
    r"(api[_ -]?key|apikey|secret|access[_ -]?token|auth[_ -]?token|_token\b|token|"
    r"password|passwd|pwd|credential|client[_ -]?secret|private[_ -]?key|login|"
    r"session[_ -]?id)",
    re.IGNORECASE,
)

# Facet value TYPES a corpus may declare. These drive both COERCION (a value is validated/cast to
# its facet type before storage) and QUERYABILITY (numeric facets support sum/avg; all support
# count/group_by/list/lookup). Corpus-neutral — an adapter picks the type per facet.
FACET_TYPES: frozenset[str] = frozenset({"text", "int", "real", "bool", "text[]"})


@dataclass(frozen=True)
class FacetSpec:
    """One adapter-DECLARED structured facet (#36/#40): the positive, fail-closed allowlist entry
    that makes a field both WRITABLE (a harvested fact for it is stored) and QUERYABLE (aggregate
    validates against declared facets, not a dataclass). The core knows NO facet name; the adapter
    declares them.

    name        — the facet/field name (the StructuredFact.field it accepts).
    type        — one of FACET_TYPES; drives value coercion + which aggregations are honest.
    description — a short human note (surfaced in observability; optional).
    """

    name: str
    type: str = "text"
    description: str = ""

    def __post_init__(self):
        if not (self.name and self.name.strip()):
            raise ValueError("FacetSpec.name must be a non-empty string")
        if self.type not in FACET_TYPES:
            raise ValueError(
                f"FacetSpec {self.name!r}: type {self.type!r} not in {sorted(FACET_TYPES)}")


def coerce_facet_value(value: object, facet_type: str) -> object:
    """Validate + coerce a fact value to its declared facet type. Raises ValueError on a value that
    can't be represented as that type (a MISTYPED fact — the caller degrades that ONE facet, never
    crashing the batch). Returns the coerced value (JSON-serialisable for storage)."""
    if value is None:
        return None
    if facet_type == "text":
        return str(value)
    if facet_type == "int":
        if isinstance(value, bool):  # bool is an int subclass; reject to avoid silent True==1
            raise ValueError(f"expected int, got bool {value!r}")
        return int(value)
    if facet_type == "real":
        if isinstance(value, bool):
            raise ValueError(f"expected real, got bool {value!r}")
        return float(value)
    if facet_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        raise ValueError(f"expected bool, got {value!r}")
    if facet_type == "text[]":
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        raise ValueError(f"expected list for text[], got {type(value).__name__}")
    raise ValueError(f"unknown facet type {facet_type!r}")  # pragma: no cover


@dataclass(frozen=True)
class StructuredFact:
    """One structured fact an adapter lifts for the metadata sidecar.

    Fields:
      entity_id   — the project/entity this fact is about (the adapter's project_id). The core
                    groups facts onto the matching sidecar record by this + source_set.
      field       — the sidecar column/slot name this value fills. The core treats it as an
                    opaque string; it never enumerates the legal set (that's the adapter's + the
                    sidecar schema's business).
      value       — the clean scalar value (str/int/float/bool) or a small list. NEVER a secret
                    (the harvester enforces that before a fact is ever constructed).
      provenance  — where/how this value was obtained + its trust level, kept generic. e.g.
                    "descriptor" (a system-of-record field, authoritative) vs "derived" (best-
                    effort constructed, flagged so a consumer knows it isn't a verbatim literal).
    """

    entity_id: str
    field: str
    value: object
    provenance: str = "descriptor"


def looks_secret(key: str) -> bool:
    """True if `key` matches the credential-adjacent secret-key shape (corpus-neutral)."""
    return bool(_SECRET_KEY_RE.search(key))


class FieldWhitelistHarvester:
    """Reusable, security-hardened primitive: lift a WHITELIST of clean fields from a parsed
    key→value mapping, refusing secrets fail-closed. Knows NO concrete field name — the adapter
    passes the whitelist, so the same primitive serves any corpus's descriptor schema.

    `whitelist` maps a SOURCE key (as it appears in the descriptor) → the sidecar FIELD name the
    value should fill. Only these keys are read; everything else (every secret block) is excluded
    by construction. A value whose SOURCE key looks secret is refused even if whitelisted (so a
    renamed upstream field colliding with a whitelisted name can never be stored).
    """

    def __init__(self, whitelist: dict[str, str]):
        self._whitelist = dict(whitelist)

    def lift(self, source: dict) -> dict[str, str]:
        """Return {sidecar_field: clean_value} for the whitelisted, non-secret, non-empty keys.

        Pure + deterministic (no I/O). A non-dict `source`, a missing key, a non-scalar value, an
        empty/placeholder value, or a secret-looking key each yields no entry for that field."""
        out: dict[str, str] = {}
        if not isinstance(source, dict):
            return out
        for src_key, field_name in self._whitelist.items():
            cleaned = self._clean_str(src_key, source.get(src_key))
            if cleaned is not None:
                out[field_name] = cleaned
        return out

    @staticmethod
    def _clean_str(key: str, value: object) -> str | None:
        """Coerce a whitelisted scalar to a clean string, or None. Refuses non-scalars, empties,
        redaction placeholders, and (belt-and-braces) any value whose KEY looks secret."""
        if value is None or isinstance(value, (dict, list)):
            return None
        if looks_secret(key):
            return None
        s = str(value).strip().strip("'\"").strip()
        if not s or s.startswith("[REDACTED"):
            return None
        return s
