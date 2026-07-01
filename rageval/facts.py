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
_SECRET_KEY_RE = re.compile(r"(_key|_token|secret|password|api_key|session_id)", re.IGNORECASE)


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
