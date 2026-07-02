"""Shared test helpers.

`make_record` builds a ProjectRecord while routing the adapter-FACET kwargs (app_name/domain/…)
into the generic `facts` dict — those fields are no longer hardcoded core columns after the #36
schema-agnostic-store refactor; they are the sample adapter's declared facets. This keeps the many
existing fixtures terse while honouring the new store shape.
"""

from __future__ import annotations

from rageval.sidecar import ProjectRecord

# The adapter-fact facet names that used to be hardcoded ProjectRecord columns and are now the
# SAMPLE adapter's declared facets (schema-agnostic store).
_FACET_KWARGS = frozenset({
    "app_name", "domain", "landing_url", "app_number", "bundle_id", "localization",
    "contact_emails",
})


def make_record(**kwargs) -> ProjectRecord:
    """Build a ProjectRecord, routing any adapter-FACET kwarg into `facts` (generic columns pass
    through unchanged)."""
    facts = dict(kwargs.pop("facts", {}) or {})
    for name in list(kwargs):
        if name in _FACET_KWARGS:
            facts[name] = kwargs.pop(name)
    rec = ProjectRecord(**kwargs)
    for name, value in facts.items():
        rec.facts[name] = value
    return rec
