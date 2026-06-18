"""Pluggable PII DETECTION backend — the *detector* under the policy-aware PII layer.

WHY this module exists (and why it is SEPARATE from the policy in redact.py):
`redact.py` owns the POLICY — the keep-or-redact decision (keep PUBLISHED / role-based
contacts, redact PERSONAL data). That policy is backend-AGNOSTIC and must NOT change when
we swap detectors. This module owns the DETECTOR — the thing that finds PII spans and labels
them with an entity type (EMAIL_ADDRESS, PERSON, PHONE_NUMBER, IBAN, CREDIT_CARD, …). The
policy sits ABOVE the detector and decides, per span, keep vs redact.

    PII POLICY layer (redact.py: keep published/role-based, redact personal)   ← unchanged
            │
       ┌────┴───────────────────────────────┐
     regex backend (DEFAULT)            presidio backend (OPTIONAL)
     light, no model download           NER: EMAIL_ADDRESS, PERSON, PHONE_NUMBER, IBAN, …

Two backends, ONE interface — mirroring the embeddings local/openai factory (get_embedder):
  * regex    — the lightweight, dependency-free detector (emails only, by design). DEFAULT.
               Zero heavy deps, no model download → the public demo and the fast test suite
               run out of the box.
  * presidio — Microsoft Presidio's AnalyzerEngine (spaCy-backed NER). Optional extra
               (`pip install -e ".[pii]"` + a spaCy model). A RICHER detector: it labels
               PERSON / PHONE_NUMBER / IBAN / CREDIT_CARD spans the regex backend never sees.
               It does NOT make keep/redact decisions — it only proposes labelled spans;
               redact.py's policy still decides. (Presidio's own `allow_list` is used as a
               fast pre-filter for role-based local-parts, but the authoritative keep/redact
               call stays in the policy layer.)

A detector returns a list of PiiSpan(start, end, entity_type). The policy layer then walks
the spans, applies the keep/redact rule per entity_type, and substitutes the readable
placeholder. This keeps detection and policy cleanly separated, exactly like the embeddings
backends keep "how we vectorise" separate from "what we do with the vectors".
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol

# --- Entity types (Presidio's vocabulary; the regex backend emits the EMAIL one) ----------
# We standardise on Presidio's entity-type names so the policy layer reasons about a single
# vocabulary regardless of which detector produced the span.
ENTITY_EMAIL = "EMAIL_ADDRESS"
ENTITY_PERSON = "PERSON"
ENTITY_PHONE = "PHONE_NUMBER"
ENTITY_IBAN = "IBAN_CODE"
ENTITY_CREDIT_CARD = "CREDIT_CARD"

# The richer set Presidio is asked to detect. Kept here so it's documented in one place and
# easy to extend. (The regex backend only ever emits EMAIL_ADDRESS.)
PRESIDIO_ENTITIES = (
    ENTITY_EMAIL,
    ENTITY_PERSON,
    ENTITY_PHONE,
    ENTITY_IBAN,
    ENTITY_CREDIT_CARD,
)

# Readable placeholders, one per entity type. The policy layer maps entity_type → placeholder
# so a redacted span reads as `[REDACTED_PERSON]` / `[REDACTED_PHONE]` etc. — same human-
# readable style as the regex backend's `[REDACTED_EMAIL]`.
PLACEHOLDERS: dict[str, str] = {
    ENTITY_EMAIL: "[REDACTED_EMAIL]",
    ENTITY_PERSON: "[REDACTED_PERSON]",
    ENTITY_PHONE: "[REDACTED_PHONE]",
    ENTITY_IBAN: "[REDACTED_IBAN]",
    ENTITY_CREDIT_CARD: "[REDACTED_CREDIT_CARD]",
}


@dataclass(frozen=True)
class PiiSpan:
    """A detected PII span: half-open [start, end) char offsets + the entity label.

    `local_part` is set for EMAIL_ADDRESS spans (the bit before '@'), because the policy's
    role-based keep rule (`support@`, `info@`, …) needs it. It's None for non-email entities.
    """

    start: int
    end: int
    entity_type: str
    local_part: str | None = None


class PiiDetector(Protocol):
    """Anything that finds PII spans in text and labels them. Detection ONLY — the keep/redact
    decision lives in the policy layer (redact.py), not here."""

    def detect(self, text: str, *, allow_list: frozenset[str] = frozenset()) -> list[PiiSpan]:
        """Return PII spans in `text`. `allow_list` is an OPTIONAL detector-side hint of values
        to skip (e.g. role local-parts a backend can cheaply pre-filter); the policy layer is
        still authoritative and re-checks every returned span."""
        ...


# --- Regex backend (DEFAULT) ---------------------------------------------------------------
# A standalone email address. Intentionally NARROW (high precision): local-part @ domain with a
# real TLD — so we don't false-positive on prices, ids, version numbers, or stray "@handle"
# mentions. The negative lookahead `(?!:\S)` skips an email immediately followed by ":<password>"
# — that shape is a credential handled by redact_secrets BEFORE the PII pass runs.
_EMAIL_RE = re.compile(
    r"\b(?P<local>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b(?!:\S)"
)


class RegexPiiDetector:
    """The lightweight, dependency-free detector. Emails only — by design (see redact.py for
    why phone/person detection needs locale-aware/NER parsing we don't fake in the regex path).
    This is the DEFAULT backend: no model download, fast, deterministic, fully unit-testable."""

    def detect(self, text: str, *, allow_list: frozenset[str] = frozenset()) -> list[PiiSpan]:
        spans: list[PiiSpan] = []
        for m in _EMAIL_RE.finditer(text):
            local = m.group("local")
            # allow_list pre-filter: a role local-part the policy would keep anyway. Skipping it
            # here is a cheap optimisation; the policy layer remains authoritative.
            if local.lower() in allow_list:
                continue
            spans.append(PiiSpan(m.start(), m.end(), ENTITY_EMAIL, local_part=local))
        return spans


# --- Presidio backend (OPTIONAL) -----------------------------------------------------------
class PresidioPiiDetector:
    """Microsoft Presidio's AnalyzerEngine as a richer NER detector.

    Optional: requires the `[pii]` extra (`presidio-analyzer`, `presidio-anonymizer`) AND a
    spaCy model (`python -m spacy download en_core_web_sm`). Selected via
    RAGEVAL_PII_BACKEND=presidio. The AnalyzerEngine is loaded LAZILY and cached on the
    instance — the (slow) spaCy load happens once per process, mirroring LocalEmbedder.

    Detection ONLY: it returns labelled spans. The policy layer (redact.py) still decides
    keep vs redact for each span. We DO pass the role local-parts as Presidio's native
    `allow_list` so role-based emails are pre-filtered, but the policy re-checks regardless."""

    def __init__(self, spacy_model: str):
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
        except ImportError as e:  # pragma: no cover - exercised only when extra is absent
            raise RuntimeError(
                "RAGEVAL_PII_BACKEND=presidio requires the presidio packages and a spaCy model. "
                'Install them with:  pip install -e ".[pii]"  &&  '
                f"python -m spacy download {spacy_model}"
            ) from e

        # Build the NLP engine bound to the configured spaCy model (default en_core_web_sm;
        # en_core_web_lg is the higher-accuracy production option — see README). If the model
        # isn't downloaded, spaCy raises a clear OSError telling you to download it.
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        )
        nlp_engine = provider.create_engine()
        self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        self._entities = list(PRESIDIO_ENTITIES)

    def detect(self, text: str, *, allow_list: frozenset[str] = frozenset()) -> list[PiiSpan]:
        if not text:
            return []
        results = self._analyzer.analyze(
            text=text,
            language="en",
            entities=self._entities,
            # Presidio's native allow_list: substrings it will NOT flag. We feed the role
            # local-parts so role-based emails are pre-filtered at detection time. The policy
            # layer still re-checks every returned span (defence-in-depth / single source of truth).
            allow_list=list(allow_list) if allow_list else None,
        )
        spans: list[PiiSpan] = []
        for r in results:
            entity = r.entity_type
            if entity not in PLACEHOLDERS:
                continue  # only entities we have a placeholder/policy for
            local = None
            if entity == ENTITY_EMAIL:
                # Recover the local-part from the matched span so the policy's role-based keep
                # rule works identically to the regex backend.
                matched = text[r.start:r.end]
                local = matched.split("@", 1)[0] if "@" in matched else matched
            spans.append(PiiSpan(r.start, r.end, entity, local_part=local))
        return spans


# --- Factory (mirrors get_embedder) --------------------------------------------------------
def _pii_backend() -> str:
    """The selected PII backend: RAGEVAL_PII_BACKEND (`regex` DEFAULT | `presidio`)."""
    return (os.environ.get("RAGEVAL_PII_BACKEND") or "regex").strip().lower() or "regex"


def _spacy_model() -> str:
    """The spaCy model the presidio backend loads (default `en_core_web_sm`; configurable —
    document `en_core_web_lg` as the production option)."""
    return (os.environ.get("RAGEVAL_PII_SPACY_MODEL") or "en_core_web_sm").strip() or "en_core_web_sm"


def get_pii_detector(backend: str | None = None, *, spacy_model: str | None = None) -> PiiDetector:
    """Factory: build the configured PII DETECTOR. Mirrors get_embedder().

    `backend` defaults to RAGEVAL_PII_BACKEND (regex). Build once and reuse — constructing the
    presidio detector loads a spaCy model into memory. The regex backend is cheap to construct.
    """
    backend = (backend or _pii_backend()).lower()
    if backend == "presidio":
        return PresidioPiiDetector(spacy_model or _spacy_model())
    if backend not in ("regex", ""):
        raise ValueError(
            f"Unknown RAGEVAL_PII_BACKEND={backend!r}. Use 'regex' (default) or 'presidio'."
        )
    return RegexPiiDetector()
