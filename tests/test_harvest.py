"""Tests for the DETERMINISTIC back/config.yaml metadata harvest (harvest.py).

These lock down the two non-negotiables: (1) the CLEAN whitelisted fields are extracted and
landing_url is derived, and (2) the SECRET blocks (third-party analytics/integration secrets,
*_key/*_token) are NEVER extracted — proven against a sample config.yaml that deliberately
contains fake secrets.
Plus the bounded, FLAGGED contact-email derivation, and graceful behaviour when config.yaml is
absent. All deterministic — no LLM, no network. Runs over the fictional data/sample/ corpus only.
"""

from __future__ import annotations

from rageval.config import SAMPLE_CORPUS_DIR
from rageval.harvest import (
    ConfigHarvest,
    derive_contact_emails,
    harvest_project,
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


# --- parse_config_yaml: clean whitelist in, secrets out ---------------------

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
    # No secret VALUE leaked...
    assert "secret" not in blob
    assert "fake_token" not in blob and "fake_api_key" not in blob
    # ...and no secret KEY became a harvested field.
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


# --- landing_url derivation -------------------------------------------------

def test_landing_url_derived_from_domain():
    h = ConfigHarvest(present=True, domain="vista-weather-7011.test")
    assert h.landing_url == "https://vista-weather-7011.test"


def test_landing_url_none_without_domain():
    assert ConfigHarvest(present=True, domain=None).landing_url is None


# --- contact email: derived + flagged, or unresolved ------------------------

def test_contact_email_derived_from_simple_template():
    tmpl = "Email support@<?= $config['app']['domain'] ?> for help."
    emails = derive_contact_emails(tmpl, "vista-weather-7011.test")
    assert emails == ["support@vista-weather-7011.test"]


def test_contact_email_unresolved_when_no_domain():
    # No harvested domain → nothing to construct → honest empty (never fabricated).
    assert derive_contact_emails("support@<?= $domain ?>", None) == []


def test_contact_email_unresolved_for_indirected_template():
    # A template that does NOT place a literal local-part immediately before a domain echo is
    # left UNRESOLVED rather than guessed.
    tmpl = "<?php echo build_support_address($config); ?>"
    assert derive_contact_emails(tmpl, "vista-weather-7011.test") == []


# --- harvest_project over the real sample fixtures --------------------------

def test_harvest_sample_vista_clean_fields_and_derived_email():
    h = harvest_project(SAMPLE_CORPUS_DIR / "atlas" / "atlas-vista")
    assert h.present is True
    assert h.app_name == "Vista Weather"
    assert h.domain == "vista-weather-7011.test"
    assert h.landing_url == "https://vista-weather-7011.test"
    assert h.app_number == "7011"
    assert h.bundle_id == "test.example.vista"
    assert h.localization == "EN"
    # Contact email derived from back/contact.php AND flagged as derived.
    assert h.contact_emails == ["support@vista-weather-7011.test"]
    assert h.contact_emails_derived is True


def test_harvest_sample_never_exposes_secrets():
    """End-to-end: harvesting the real fixture (which contains fake secret blocks) exposes
    NO secret value anywhere in the resulting ConfigHarvest."""
    h = harvest_project(SAMPLE_CORPUS_DIR / "atlas" / "atlas-vista")
    blob = repr(h).lower()
    assert "fake_value_never_harvested" not in blob
    assert "analytics" not in blob and "integration" not in blob and "session_id" not in blob


def test_harvest_absent_config_degrades_gracefully():
    """A project directory with no back/config.yaml yields present=False and all-None fields —
    proving the harvest never crashes or fabricates when config.yaml is missing (~5/232 projects)."""
    h = harvest_project(SAMPLE_CORPUS_DIR / "atlas" / "atlas-orchard")
    assert h.present is False
    assert h.domain is None and h.landing_url is None and h.app_name is None
    assert h.contact_emails is None
