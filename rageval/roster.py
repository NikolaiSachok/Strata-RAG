"""Roster loader — DETERMINISTIC project-id → publisher join (NOT the LLM).

WHY this module exists (the definitional split it implements):
  The engine has TWO distinct "name" facets that were historically conflated into one field:

    * app_name  — the project's PRODUCT / display name (e.g. a fictional "Frostline Fishing
                  Log"). It is stated in the docs / config.yaml, so the LLM (or the deterministic
                  config.yaml harvest) extracts it confidently.
    * publisher — an AUTHORITATIVE label each project is associated with, supplied via a roster
                  TSV (project-id → publisher). It may DIFFER from the product title, and may be
                  ABSENT from the project's own docs, so NO amount of LLM reading can recover it
                  reliably — it lives in a hand-maintained roster TSV (the ground truth).

  They COINCIDE only when the publisher happens to also be the app name. For everything else,
  asking the LLM for the publisher would mean asking it to guess — so we DON'T. We do a
  deterministic TSV join instead, and trust the roster (ground truth) over LLM inference.

GENERIC FEATURE / OPTIONAL DATA:
  This loader is GENERIC. When a roster file is absent (a corpus with no roster, or a project
  not listed in it), `publisher` is simply None — graceful degradation, identical code path.
  The shipped sample corpus uses FICTIONAL sample rosters (`data/sample/*.tsv`).

The join key is the LEADING NUMERIC id parsed out of the engine's `project_id` — robust to
the messy id shapes a real corpus can contain (`2268`, `1490-sp08`, `2288_Summit (...)`,
zero-padded `0018`). The TSV is the authoritative `№ / ID / Publisher / Bundle` mapping; we
read only the `№` (id) and `Publisher` columns.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .config import ROSTER_DIR, SAMPLE_ROSTER_DIR, SETTINGS, Settings, is_sample_corpus

# A source_set maps to a roster FILE by its base FAMILY (the part before the first '-'):
# every `northwind*` set shares one `northwind.tsv`, every `atlas*` set shares one `atlas.tsv`.
# Add a family = add one line here; nothing else changes. This is the ONLY place the
# family→file mapping lives. An external overlay extends it via the PUBLIC `register_family()`
# API below (so it never mutates this dict directly) — see register_family.
_FAMILY_TO_TSV_STEM: dict[str, str] = {
    # --- synthetic sample corpus (fictional ids/publishers) ---
    "northwind": "northwind",
    "atlas": "atlas",
}


def register_family(family: str, tsv_stem: str) -> None:
    """Register an external source-set `family` → its roster TSV `tsv_stem` (PUBLIC API).

    This is the roster-side companion to `register_adapter()`: an overlay that adds a corpus
    family (e.g. a `mycorp` adapter) calls `register_family("mycorp", "mycorp")` so a project
    in that family joins against `<roster_dir>/mycorp.tsv` for its authoritative publisher —
    WITHOUT mutating the private `_FAMILY_TO_TSV_STEM` dict directly.

    `family` is matched the same way the loader resolves a source_set: case-insensitively
    against the part of the source_set before the first '-' (so `mycorp`, `mycorp-extra`, … all
    share one TSV). `tsv_stem` is the roster file's basename without extension (`mycorp` →
    `mycorp.tsv`, looked up in the active roster dir). Idempotent/last-wins: re-registering a
    family replaces its stem. Both args must be non-empty strings (a cheap guard against
    mis-wiring the seam)."""
    if not (isinstance(family, str) and family.strip()):
        raise ValueError(f"register_family expects a non-empty family name, got {family!r}")
    if not (isinstance(tsv_stem, str) and tsv_stem.strip()):
        raise ValueError(f"register_family expects a non-empty tsv_stem, got {tsv_stem!r}")
    _FAMILY_TO_TSV_STEM[family.strip().lower()] = tsv_stem.strip()

# Leading numeric id out of a project_id: "2268"→2268, "northwind/2268"→2268 (the source_set
# prefix is stripped by callers, but be defensive), "1490-sp08"→1490, "2288_Summit"→2288,
# "0018"→18. We take the FIRST run of digits; everything after it (suffix/name) is ignored.
_LEADING_ID_RE = re.compile(r"(\d+)")


def extract_numeric_id(project_id: str) -> int | None:
    """Parse the leading numeric id from a (possibly suffixed) project_id.

    Returns an int (so "0018" and "18" join to the same row), or None when the id carries no
    digits at all (e.g. an `atlas-ledger`-style named project — those simply have no roster row)."""
    if not project_id:
        return None
    m = _LEADING_ID_RE.search(project_id)
    return int(m.group(1)) if m else None


def _tsv_stem_for(source_set: str) -> str | None:
    """Map a source_set to its roster TSV stem via the base family (text before first '-')."""
    family = source_set.split("-", 1)[0].lower()
    return _FAMILY_TO_TSV_STEM.get(family)


def _roster_dir_for(settings: Settings) -> Path:
    """Which directory holds the roster TSVs for THIS run.

    Sample corpus → the sample rosters live next to it (`data/sample/`); any other corpus root →
    the top-level `data/` rosters. This keeps the synthetic demo self-contained and the loader
    generic (no hard-coded paths). Overridable via RAGEVAL_ROSTER_DIR (see config)."""
    if settings.roster_dir:
        return Path(settings.roster_dir).expanduser()
    # Same sample-vs-custom predicate the collection-name suffix uses (config.is_sample_corpus),
    # so "this is the sample corpus" means one thing across the engine.
    return SAMPLE_ROSTER_DIR if is_sample_corpus(settings.corpus_root) else ROSTER_DIR


def _load_tsv(path: Path) -> dict[int, str]:
    """Parse one roster TSV (`№ / ID / Publisher / Bundle`) → {numeric_id: publisher}.

    Keys on the `№` column (the project folder id); reads only `№` and `Publisher`. Tolerant of a
    missing cell, blank rows, and ids with a zero-pad. A malformed/absent file yields {}."""
    out: dict[int, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    reader = csv.reader(text.splitlines(), delimiter="\t")
    next(reader, None)  # skip the "№ ID Publisher Bundle" header
    for cols in reader:
        if not cols or not cols[0].strip():
            continue
        nid = extract_numeric_id(cols[0].strip())
        publisher = cols[2].strip() if len(cols) > 2 else ""
        if nid is not None and publisher:
            out.setdefault(nid, publisher)
    return out


class Roster:
    """Lazy, cached project-id → publisher resolver over per-family roster TSVs.

    Construct once per ingest/enrich pass and call `publisher(source_set, project_id)` per
    project. TSV files are loaded on first use and memoised. A MISSING roster file (a corpus with
    no roster, or a family with no roster) and a MISSING row both resolve to None — never an
    error, never a guess. This is the deterministic, authoritative ground-truth side of the
    title-vs-publisher split."""

    def __init__(self, roster_dir: Path):
        self._dir = roster_dir
        self._cache: dict[str, dict[int, str]] = {}  # tsv stem → {id: publisher}

    @classmethod
    def for_settings(cls, settings: Settings = SETTINGS) -> "Roster":
        return cls(_roster_dir_for(settings))

    def _map_for_stem(self, stem: str) -> dict[int, str]:
        if stem not in self._cache:
            self._cache[stem] = _load_tsv(self._dir / f"{stem}.tsv")
        return self._cache[stem]

    def publisher(self, source_set: str, project_id: str) -> str | None:
        """The authoritative publisher for one project, or None.

        None when: the source_set has no mapped family, the roster file is absent, the project_id
        has no numeric id, or the id isn't a row in the roster. All are normal, expected outcomes —
        the roster is intentionally allowed to be incomplete (not every project id need be listed)
        or absent entirely."""
        stem = _tsv_stem_for(source_set)
        if stem is None:
            return None
        nid = extract_numeric_id(project_id)
        if nid is None:
            return None
        return self._map_for_stem(stem).get(nid)
