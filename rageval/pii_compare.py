"""PII backend COMPARISON harness — run regex vs Presidio over the SAME corpus, side by side.

WHY this is a teaching/portfolio piece: the engine's PII layer is built so the keep/redact
POLICY is backend-AGNOSTIC and only the DETECTOR swaps (see pii.py / redact.py). This harness
makes that concrete and measurable — it runs BOTH detectors over the same documents, applies
the SAME policy to each, and reports where they AGREE and (the interesting part) where they
DISAGREE:

  * regex-only catches    — emails the lightweight detector found that Presidio didn't.
  * presidio-only catches — PERSON / PHONE_NUMBER / IBAN / CREDIT_CARD spans the regex detector
                            is STRUCTURALLY blind to (free-text names, locale-y phone shapes, …).

The takeaway it demonstrates: cheap deterministic regex = high precision on STRUCTURED patterns
(emails), zero deps, but blind to names/free-text PII; Presidio NER = broad entity coverage
(catches names/phones/IBANs) at the cost of a heavier, probabilistic backend (false positives
possible). The policy layer treats both identically — which is the whole point.

Run it:
    python -m rageval.pii_compare                 # sample corpus (default)
    RAGENGINE_CORPUS_ROOT=/path python -m rageval.pii_compare   # a real corpus

Degrades gracefully: if Presidio isn't installed it reports the regex side and tells you how to
enable the comparison — it never crashes (so the core suite stays green without the extra).

LEAK SAFETY: the report NEVER prints a raw PII value. Every span is shown MASKED (first char +
length + entity type + offset + which backend + the keep/redact decision), so the harness is safe
to run on a real corpus and paste into notes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .classify import CorpusRules, classify
from .config import SETTINGS, Settings
from .pii import (
    PRESIDIO_ENTITIES,
    PiiDetector,
    PiiSpan,
    RegexPiiDetector,
    get_pii_detector,
)
from .redact import PiiPolicy, _keep_span, redact_secrets
from .sources import discover_all


def mask_value(value: str) -> str:
    """Mask a raw PII value for safe display: first char + a length-bucketed shape, never the
    full string. e.g. "jordan.lee@example.com" → "j…[len=22]". Empty → "∅"."""
    value = value or ""
    if not value:
        return "∅"
    return f"{value[0]}…[len={len(value)}]"


@dataclass(frozen=True)
class FoundSpan:
    """A detected span, recorded with its doc, masked value, and the policy's keep/redact call."""

    doc_id: str
    doc_type: str
    entity_type: str
    start: int
    end: int
    masked: str
    redact: bool  # True → policy says redact; False → policy says keep (published/role-based)


def _detect_in_text(detector: PiiDetector, text: str, doc_id: str, doc_type: str,
                    policy: PiiPolicy) -> list[FoundSpan]:
    """Run a detector over already-secret-redacted text and tag each span with the policy call."""
    out: list[FoundSpan] = []
    for s in detector.detect(text, allow_list=policy.role_local_parts):
        keep = _keep_span(s, doc_type, policy)
        out.append(FoundSpan(
            doc_id=doc_id, doc_type=doc_type, entity_type=s.entity_type,
            start=s.start, end=s.end, masked=mask_value(text[s.start:s.end]),
            redact=not keep,
        ))
    return out


def _span_key(f: FoundSpan) -> tuple:
    """Identity for agreement matching: same doc + overlapping-ish offset + entity type. We key on
    the exact (doc, start, entity) — the regex and Presidio offsets for a shared email coincide."""
    return (f.doc_id, f.start, f.entity_type)


@dataclass
class Comparison:
    """Result of comparing the two backends over one corpus."""

    regex_spans: list[FoundSpan]
    presidio_spans: list[FoundSpan] | None  # None → Presidio unavailable (not installed)
    presidio_error: str = ""

    @property
    def presidio_available(self) -> bool:
        return self.presidio_spans is not None


def compare(settings: Settings = SETTINGS) -> Comparison:
    """Discover + classify the corpus, then run BOTH detectors (regex always; Presidio if
    installed) over every INCLUDED doc's secret-redacted text, applying the shared policy."""
    rules = CorpusRules.load()
    policy = rules.pii_policy
    docs = discover_all(settings.corpus_root)

    # Build each detector ONCE. Regex always works. Presidio may be absent → degrade gracefully.
    regex_det: PiiDetector = RegexPiiDetector()
    presidio_det: PiiDetector | None = None
    presidio_error = ""
    try:
        presidio_det = get_pii_detector("presidio", spacy_model=settings.pii_spacy_model)
    except Exception as e:  # not installed, or spaCy model missing
        presidio_error = str(e).splitlines()[0]

    regex_spans: list[FoundSpan] = []
    presidio_spans: list[FoundSpan] | None = [] if presidio_det is not None else None

    for doc in docs:
        if not classify(doc, rules).include:
            continue
        doc_id = f"{doc.source_set}/{doc.project_id}/{doc.doc_path.name}"
        # Mirror the real pipeline: secret redaction runs FIRST (so an email:password blob is a
        # credential, not a bare email), THEN PII detection runs on what remains.
        clean, _ = redact_secrets(doc.raw_text)
        regex_spans.extend(_detect_in_text(regex_det, clean, doc_id, doc.doc_type, policy))
        if presidio_det is not None:
            presidio_spans.extend(  # type: ignore[union-attr]
                _detect_in_text(presidio_det, clean, doc_id, doc.doc_type, policy))

    return Comparison(regex_spans=regex_spans, presidio_spans=presidio_spans,
                      presidio_error=presidio_error)


def _counts_by_entity(spans: list[FoundSpan]) -> dict[str, tuple[int, int]]:
    """entity_type → (n_redact, n_keep)."""
    out: dict[str, list[int]] = {}
    for s in spans:
        slot = out.setdefault(s.entity_type, [0, 0])
        slot[0 if s.redact else 1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def render(cmp: Comparison) -> str:
    """Render the human-readable comparison report (all PII values masked)."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("PII BACKEND COMPARISON — regex (default) vs Presidio (optional NER)")
    lines.append("=" * 78)
    lines.append("Policy is backend-agnostic: the SAME keep/redact rule is applied to both.")
    lines.append("All PII values are MASKED (first char + length); raw values are never printed.")
    lines.append("")

    def _entity_block(title: str, spans: list[FoundSpan]) -> None:
        lines.append(f"[{title}]  total spans: {len(spans)}")
        counts = _counts_by_entity(spans)
        if not counts:
            lines.append("  (none)")
        for entity in sorted(counts):
            n_red, n_keep = counts[entity]
            lines.append(f"  {entity:<14} redact={n_red:<3} keep={n_keep:<3}")
        lines.append("")

    _entity_block("REGEX backend", cmp.regex_spans)

    if not cmp.presidio_available:
        lines.append("[PRESIDIO backend]  UNAVAILABLE")
        lines.append(f"  Presidio not usable: {cmp.presidio_error or 'not installed'}")
        lines.append('  Install to compare:  pip install -e ".[pii]"  '
                     "&&  python -m spacy download en_core_web_sm")
        lines.append("")
        lines.append("Comparison (agreement/disagreement) needs both backends — regex side shown above.")
        lines.append("=" * 78)
        return "\n".join(lines)

    presidio_spans = cmp.presidio_spans or []
    _entity_block("PRESIDIO backend", presidio_spans)

    # Agreement / disagreement by (doc, start, entity_type).
    regex_keys = {_span_key(s): s for s in cmp.regex_spans}
    presidio_keys = {_span_key(s): s for s in presidio_spans}
    agree = sorted(regex_keys.keys() & presidio_keys.keys())
    regex_only = sorted(regex_keys.keys() - presidio_keys.keys())
    presidio_only = sorted(presidio_keys.keys() - regex_keys.keys())

    lines.append(f"AGREEMENT (both flagged the same span): {len(agree)}")
    for k in agree:
        s = regex_keys[k]
        decision = "REDACT" if s.redact else "KEEP"
        lines.append(f"  {s.entity_type:<14} {s.masked:<16} @{s.start:<5} {decision:<6} {s.doc_id}")
    lines.append("")

    lines.append(f"REGEX-ONLY (regex caught, Presidio missed): {len(regex_only)}")
    for k in regex_only:
        s = regex_keys[k]
        decision = "REDACT" if s.redact else "KEEP"
        lines.append(f"  {s.entity_type:<14} {s.masked:<16} @{s.start:<5} {decision:<6} {s.doc_id}")
    lines.append("")

    lines.append(f"PRESIDIO-ONLY (NER caught, regex structurally blind): {len(presidio_only)}")
    for k in presidio_only:
        s = presidio_keys[k]
        decision = "REDACT" if s.redact else "KEEP"
        lines.append(f"  {s.entity_type:<14} {s.masked:<16} @{s.start:<5} {decision:<6} {s.doc_id}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("TRADEOFF: regex = cheap, deterministic, zero-dep, high precision on STRUCTURED")
    lines.append("patterns (emails) — but blind to names/free-text PII. Presidio NER = broad entity")
    lines.append(f"coverage ({', '.join(PRESIDIO_ENTITIES)}), catches names/phones/IBANs — but heavier")
    lines.append("and probabilistic (false positives possible). The policy layer treats both the same.")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the regex and Presidio PII detection backends over the corpus."
    )
    parser.add_argument("--corpus-root", default=None,
                        help="Override the corpus root (defaults to RAGENGINE_CORPUS_ROOT / sample).")
    args = parser.parse_args(argv)

    settings = SETTINGS
    if args.corpus_root:
        import dataclasses
        from pathlib import Path
        settings = dataclasses.replace(settings, corpus_root=Path(args.corpus_root).expanduser())

    print(render(compare(settings)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
