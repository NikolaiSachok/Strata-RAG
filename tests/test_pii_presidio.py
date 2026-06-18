"""Tests for the PLUGGABLE PII detection backend (pii.py) + the comparison harness.

The presidio backend is an OPTIONAL extra (`pip install -e ".[pii]"` + a spaCy model). These
tests SKIP when it isn't installed — mirroring the openai-extra skip pattern — so the core suite
runs green WITHOUT presidio and WITHOUT any model download. The regex-backend + factory +
harness-degrades-gracefully tests below always run (no optional deps).
"""

from __future__ import annotations

import importlib.util

import pytest

from rageval.pii import (
    ENTITY_EMAIL,
    ENTITY_PERSON,
    PLACEHOLDERS,
    RegexPiiDetector,
    get_pii_detector,
)
from rageval.redact import PiiPolicy, redact_pii

# True only when BOTH presidio packages AND a spaCy model are importable. The detector
# constructor raises if the model isn't downloaded, so we additionally try to build it.
_HAS_PRESIDIO = (
    importlib.util.find_spec("presidio_analyzer") is not None
    and importlib.util.find_spec("presidio_anonymizer") is not None
)


def _presidio_usable() -> bool:
    if not _HAS_PRESIDIO:
        return False
    try:
        get_pii_detector("presidio")
        return True
    except Exception:
        return False


presidio = pytest.mark.skipif(
    not _presidio_usable(),
    reason='presidio backend not installed/usable — `pip install -e ".[pii]"` + spaCy model',
)


# --- Always-on tests: regex backend + factory + graceful degradation ----------------------

def test_factory_default_is_regex():
    assert isinstance(get_pii_detector(), RegexPiiDetector)
    assert isinstance(get_pii_detector("regex"), RegexPiiDetector)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError):
        get_pii_detector("bogus")


def test_regex_detector_finds_email_span():
    spans = RegexPiiDetector().detect("ping jordan.lee@example.com please")
    assert len(spans) == 1
    s = spans[0]
    assert s.entity_type == ENTITY_EMAIL
    assert s.local_part == "jordan.lee"


def test_regex_detector_allow_list_prefilters_role_local_parts():
    spans = RegexPiiDetector().detect(
        "mail support@app.com now", allow_list=frozenset({"support"})
    )
    assert spans == []


def test_policy_unchanged_under_regex_backend_personal_redacted():
    # Sanity: the refactored redact_pii (now detector-backed) preserves the old behavior.
    clean, n = redact_pii("contact alex.rivera@example.com", doc_type="spec",
                          detector=RegexPiiDetector())
    assert n == 1 and PLACEHOLDERS[ENTITY_EMAIL] in clean


def test_compare_harness_degrades_without_presidio():
    """The comparison harness must NOT crash when Presidio is absent: it reports the regex side
    and flags Presidio unavailable. (Always runs — proves the core path is safe.)"""
    from rageval.pii_compare import compare, render

    cmp = compare()
    text = render(cmp)
    assert "REGEX backend" in text
    if not _presidio_usable():
        assert not cmp.presidio_available
        assert "UNAVAILABLE" in text
    # Never leaks a raw email: the masked form replaces '@…'.
    assert "jordan.lee@example.com" not in text


def test_mask_value_never_emits_raw():
    from rageval.pii_compare import mask_value

    masked = mask_value("jordan.lee@example.com")
    assert "jordan.lee@example.com" not in masked
    assert masked.startswith("j") and "len=22" in masked


# --- Presidio-backed tests (SKIP when the extra isn't installed) --------------------------

@presidio
def test_presidio_detects_person_name_regex_misses():
    """Presidio's NER flags a free-text PERSON the regex detector is structurally blind to."""
    det = get_pii_detector("presidio")
    spans = det.detect("escalate billing to Sarah Williams on the finance side.")
    assert any(s.entity_type == ENTITY_PERSON for s in spans)
    # The regex detector finds nothing here (no email).
    assert RegexPiiDetector().detect("escalate billing to Sarah Williams on the finance side.") == []


@presidio
def test_presidio_policy_keeps_published_redacts_personal():
    """The keep/redact POLICY is unchanged under Presidio: a published support@ contact is KEPT,
    a personal email in an internal doc is REDACTED."""
    det = get_pii_detector("presidio")
    # Published role-based contact in a public-facing doc → kept.
    kept, n_keep = redact_pii("Email support@app.example any time.", doc_type="description",
                              detector=det)
    assert n_keep == 0 and "support@app.example" in kept
    # Personal email in an internal doc → redacted to the readable placeholder.
    clean, n = redact_pii("owner alex.rivera@example.com", doc_type="metadata", detector=det)
    assert n >= 1 and PLACEHOLDERS[ENTITY_EMAIL] in clean
    assert "alex.rivera@example.com" not in clean


@presidio
def test_presidio_compare_reports_presidio_only_person():
    """End-to-end: the comparison harness surfaces a PRESIDIO-ONLY catch (the seeded PERSON)."""
    from rageval.pii_compare import compare, render

    cmp = compare()
    assert cmp.presidio_available
    text = render(cmp)
    assert "PRESIDIO-ONLY" in text
    assert any(s.entity_type == ENTITY_PERSON for s in (cmp.presidio_spans or []))
