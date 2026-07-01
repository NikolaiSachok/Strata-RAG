"""Query-class registry — the corpus-extensible set of reasoning shapes (#38).

WHY this exists (the Phase-4 router decouple):
  The router recognises a set of QUERY CLASSES (semantic, aggregation, lookup, hybrid) and picks
  one per question. Those four are GENERIC capabilities — any corpus needs "answer from meaning"
  and "count/group over structured facts". But a NEW corpus brings NEW reasoning shapes (e.g.
  MULTI-HOP / cross-document joins) that the four don't cover, and baking those into the core
  router would make it single-corpus.

  So the query-class set is a REGISTRY. The core ships the generic classes; a corpus registers
  ADDITIONAL classes (name + detection + execution) via the public `register_query_class()` API,
  with NO core change. The router stays DETERMINISTIC-FIRST: a registered class's `detect` is a
  deterministic rule that runs before the LLM classifier; the LLM only PROPOSES where no rule
  fires. Execution for an extra class is data the class carries, not a hardcoded core branch.

DESIGN (trust boundary unchanged): a corpus's `detect` is a pure, deterministic function
question→(confidence, slots)|None; its `execute` runs the class given the router decision +
pipeline. Both are the CORPUS's code (behind the registry), so the core router/dispatch never
enumerate a corpus-specific intent string or field target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# The generic query classes the engine ships. These are corpus-NEUTRAL capabilities (not intents
# shaped around one corpus), so they live in core. A corpus EXTENDS this set; it never edits it.
GENERIC_CLASSES: tuple[str, ...] = ("semantic", "aggregation", "lookup", "hybrid")


@dataclass(frozen=True)
class QueryClass:
    """One registrable query class: a name + optional deterministic detector + optional executor.

    name    — the route label this class emits/handles (e.g. "multi_hop"). Must not collide with a
              generic class name.
    detect  — a DETERMINISTIC rule: question → (confidence, slots) when this class applies, else
              None. Runs BEFORE the LLM classifier (deterministic-first). Optional: a class the LLM
              proposes by name needs no detector.
    execute — run the class: (question, decision, pipeline) → Answer (or None to fall back to
              semantic). Optional: a class the core already knows how to run needs no executor.
    describe— a one-line hint appended to the LLM classifier prompt so the model can PROPOSE this
              class by name. Optional.
    """

    name: str
    detect: Optional[Callable[[str], Optional[tuple[float, dict]]]] = None
    execute: Optional[Callable[..., object]] = None
    describe: str = ""


# The registry: name → QueryClass. Only EXTRA (non-generic) classes live here; the generic four
# are handled by the core router/dispatch directly. Last-wins so a re-register replaces.
_REGISTRY: dict[str, QueryClass] = {}


def register_query_class(qc: QueryClass) -> None:
    """Register an EXTRA query class (PUBLIC API, #38). Rejects a name that collides with a generic
    class (those are core-owned) or an empty name — a cheap guard against mis-wiring the seam.
    Idempotent/last-wins."""
    if not (isinstance(qc, QueryClass) and qc.name and qc.name.strip()):
        raise ValueError(f"register_query_class expects a named QueryClass, got {qc!r}")
    if qc.name in GENERIC_CLASSES:
        raise ValueError(
            f"query class {qc.name!r} collides with a generic core class; pick another name")
    _REGISTRY[qc.name] = qc


def registered_classes() -> dict[str, QueryClass]:
    """A snapshot of the currently-registered EXTRA query classes (name → QueryClass)."""
    return dict(_REGISTRY)


def valid_routes() -> tuple[str, ...]:
    """Every route the router may emit: the generic classes PLUS any registered extras. The router
    validates the LLM's proposed route against this (so a corpus-registered class can be proposed
    by name), keeping the core free of any corpus-specific route string."""
    return GENERIC_CLASSES + tuple(_REGISTRY)


def detect(question: str):
    """Run every registered class's DETERMINISTIC detector, first match wins. Returns
    (QueryClass, confidence, slots) or None. This is the deterministic-first extension point the
    router consults before falling back to the LLM classifier."""
    for qc in _REGISTRY.values():
        if qc.detect is None:
            continue
        hit = qc.detect(question)
        if hit is not None:
            confidence, slots = hit
            return qc, confidence, slots
    return None


def get(name: str) -> QueryClass | None:
    """The registered extra class for `name`, or None (a generic/unknown route)."""
    return _REGISTRY.get(name)


def describe_extras() -> str:
    """Newline-joined `- name: describe` hints for the registered extras, for the LLM classifier
    prompt (empty string when none are registered → the generic prompt is unchanged)."""
    lines = [f"- {qc.name}: {qc.describe}" for qc in _REGISTRY.values() if qc.describe]
    return "\n".join(lines)
