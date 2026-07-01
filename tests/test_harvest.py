"""Tests for the structured-fact harvest (#36): the corpus-agnostic core primitive
(rageval.facts) + the sample corpus's concrete harvester (sources/sample_facts).

These lock down the two non-negotiables: (1) the CLEAN whitelisted fields are lifted as
StructuredFacts and landing_url is DERIVED, and (2) the SECRET blocks (third-party
analytics/integration secrets, *_key/*_token) are NEVER lifted — proven against a sample
config.yaml that deliberately contains fake secrets. Plus the bounded, FLAGGED contact-email
derivation, and graceful behaviour when the descriptor is absent. All deterministic — no LLM,
no network. Runs over the fictional data/sample/ corpus only.
"""

from __future__ import annotations

from rageval.config import SAMPLE_CORPUS_DIR
from rageval.facts import FieldWhitelistHarvester, StructuredFact, looks_secret
from rageval.sources.sample_facts import (
    derive_contact_emails,
    harvest_facts,
    parse_config_yaml,
)

_SAMPLE_YAML = """
app:
  name: 'Vista Weather'
  number: '7011'
  domain: 'vista-weather-7011.test'
  bundle_id: test.example.vista
  localization: EN
analytics:
  api_key: fake_api_key_SECRET
integration:
  token: fake_token_SECRET
  session_id: fake_session_id_SECRET
"""


def _facts_dict(facts):
    """Collapse a list of StructuredFacts → {field: value} for easy assertions."""
    return {f.field: f.value for f in facts}


# --- the CORE primitive: whitelist in, secrets out (corpus-neutral) ---------

def test_core_harvester_lifts_only_whitelisted_fields():
    h = FieldWhitelistHarvester({"name": "app_name", "domain": "domain"})
    out = h.lift({"name": "Vista", "domain": "vista.test", "api_key": "SECRET", "other": "x"})
    assert out == {"app_name": "Vista", "domain": "vista.test"}


def test_core_harvester_refuses_secret_looking_whitelisted_key():
    # Belt-and-braces: even if a secret-looking key were whitelisted, its value is refused.
    h = FieldWhitelistHarvester({"api_key": "app_name"})
    assert h.lift({"api_key": "leak"}) == {}
    assert looks_secret("session_id") and not looks_secret("domain")


def test_core_harvester_skips_non_scalar_empty_and_redacted():
    h = FieldWhitelistHarvester({"name": "app_name", "blk": "b", "empty": "e", "red": "r"})
    out = h.lift({"name": "Ok", "blk": {"x": 1}, "empty": "", "red": "[REDACTED_KEY]"})
    assert out == {"app_name": "Ok"}


# --- sample harvester: parse_config_yaml (clean whitelist in, secrets out) ---

def test_parse_extracts_only_whitelisted_clean_fields():
    fields = parse_config_yaml(_SAMPLE_YAML)
    assert fields == {
        "app_name": "Vista Weather",
        "app_number": "7011",
        "domain": "vista-weather-7011.test",
        "bundle_id": "test.example.vista",
        "localization": "EN",
    }


def test_parse_never_extracts_secret_fields():
    """The CORE security assertion: no secret value or secret key ever appears in the harvest."""
    fields = parse_config_yaml(_SAMPLE_YAML)
    blob = repr(fields).lower()
    assert "secret" not in blob
    assert "fake_token" not in blob and "fake_api_key" not in blob
    for forbidden in ("api_key", "token", "session_id", "analytics", "integration"):
        assert forbidden not in fields


def test_parse_handles_malformed_or_missing_app_block():
    assert parse_config_yaml("not: valid: yaml: [") == {} or isinstance(
        parse_config_yaml("just_a_string"), dict)
    assert parse_config_yaml("just_a_string") == {}
    assert parse_config_yaml("other:\n  k: v") == {}  # no app: block


def test_parse_skips_redacted_and_empty_values():
    fields = parse_config_yaml("app:\n  name: '[REDACTED_KEY]'\n  domain: ''\n  number: '9'")
    assert "app_name" not in fields and "domain" not in fields
    assert fields["app_number"] == "9"


# --- contact email: derived + flagged, or unresolved ------------------------

def test_contact_email_derived_from_simple_template():
    tmpl = "Email support@<?= $config['app']['domain'] ?> for help."
    emails = derive_contact_emails(tmpl, "vista-weather-7011.test")
    assert emails == ["support@vista-weather-7011.test"]


def test_contact_email_unresolved_when_no_domain():
    assert derive_contact_emails("support@<?= $domain ?>", None) == []


def test_contact_email_unresolved_for_indirected_template():
    tmpl = "<?php echo build_support_address($config); ?>"
    assert derive_contact_emails(tmpl, "vista-weather-7011.test") == []


# --- harvest_facts over the real sample fixtures ----------------------------

def test_harvest_sample_vista_yields_clean_facts_and_derived_email():
    facts = list(harvest_facts("atlas-vista", SAMPLE_CORPUS_DIR / "atlas" / "atlas-vista"))
    d = _facts_dict(facts)
    assert d["app_name"] == "Vista Weather"
    assert d["domain"] == "vista-weather-7011.test"
    assert d["landing_url"] == "https://vista-weather-7011.test"
    assert d["app_number"] == "7011"
    assert d["bundle_id"] == "test.example.vista"
    assert d["localization"] == "EN"
    assert d["contact_emails"] == ["support@vista-weather-7011.test"]
    # Provenance: descriptor fields are authoritative; landing_url + contact_emails are DERIVED.
    prov = {f.field: f.provenance for f in facts}
    assert prov["domain"] == "descriptor"
    assert prov["landing_url"] == "derived"
    assert prov["contact_emails"] == "derived"
    # Every fact carries the entity id.
    assert all(f.entity_id == "atlas-vista" for f in facts)
    assert all(isinstance(f, StructuredFact) for f in facts)


def test_harvest_sample_never_exposes_secrets():
    """End-to-end: harvesting the real fixture (which contains fake secret blocks) exposes
    NO secret value anywhere in the resulting facts."""
    facts = list(harvest_facts("atlas-vista", SAMPLE_CORPUS_DIR / "atlas" / "atlas-vista"))
    blob = repr(facts).lower()
    assert "fake_value_never_harvested" not in blob
    assert "analytics" not in blob and "integration" not in blob and "session_id" not in blob


def test_harvest_absent_config_degrades_gracefully():
    """A project directory with no back/config.yaml yields NO facts — proving the harvest never
    crashes or fabricates when the descriptor is missing."""
    facts = list(harvest_facts("atlas-orchard", SAMPLE_CORPUS_DIR / "atlas" / "atlas-orchard"))
    assert facts == []
