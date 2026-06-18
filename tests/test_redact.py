"""Tests for ingest-time secret redaction.

Redaction is pure (text in → clean text + count out), so it's fully unit-testable, which
is exactly what a credential-scrubber needs: a redactor you can't test is one you can't
trust. We assert (a) secret VALUES are gone, (b) surrounding content — especially the
`Brand:` field — is PRESERVED, and (c) ordinary prose is not over-redacted.
"""

from __future__ import annotations

from rageval.redact import (
    CRED_PLACEHOLDER,
    EMAIL_PLACEHOLDER,
    KEY_PLACEHOLDER,
    PiiPolicy,
    redact,
    redact_pii,
    redact_secrets,
)


def test_settings_md_keeps_brand_redacts_key():
    text = (
        "# Settings\n"
        "Brand: Aurora Tasks\n"
        "Category: to-do list\n"
        "api_key: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6\n"
    )
    clean, n = redact_secrets(text)
    assert n >= 1
    assert "Brand: Aurora Tasks" in clean          # the reason we KEEP settings.md
    assert "Category: to-do list" in clean
    assert "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6" not in clean
    assert KEY_PLACEHOLDER in clean


def test_credential_blob_all_secrets_removed():
    blob = (
        "sportsdata.io login: bot@example.com:Sup3rSecret!\n"
        "api_key = 7e3a9c1b5d2f48069a8b7c6d5e4f3a2b1c0d9e8f\n"
        "password: hunter2hunter2\n"
        "keystore: release.jks\n"
        "keystore_password: kSpassWORD123456\n"
        "figma: https://figma.com/proto/ZyXwVu987/Onboarding\n"
    )
    clean, n = redact_secrets(blob)
    assert n >= 5
    # No raw secret survives.
    for leaked in ("bot@example.com:Sup3rSecret!", "7e3a9c1b5d2f48069a8b7c6d5e4f3a2b1c0d9e8f",
                   "hunter2hunter2", "kSpassWORD123456", "figma.com/proto/ZyXwVu987"):
        assert leaked not in clean, f"secret leaked: {leaked}"
    # The non-secret 'keystore: release.jks' reference line is preserved.
    assert "release.jks" in clean


def test_uuid_style_key_redacted_without_label():
    text = "Reference token 550e8400-e29b-41d4-a716-446655440000 appears inline."
    clean, n = redact_secrets(text)
    assert n == 1 and KEY_PLACEHOLDER in clean
    assert "550e8400-e29b-41d4-a716-446655440000" not in clean


def test_compound_key_name_is_caught():
    # A prefix before the secret word must still trip the contextual rule.
    clean, n = redact_secrets("client_secret = abc123def456ghi")
    assert n == 1 and "abc123def456ghi" not in clean


def test_login_and_password_use_credential_placeholder():
    clean, _ = redact_secrets("login: admin\npassword: letmein123")
    assert clean.count(CRED_PLACEHOLDER) == 2


def test_plain_prose_is_not_over_redacted():
    prose = (
        "Brand: Tide Tracker. Log in from the home screen to track your habits. "
        "The reward token animation is a nice touch and users love the secret cove level."
    )
    clean, n = redact_secrets(prose)
    # Mid-sentence 'log in' / 'token' / 'secret' are NOT key:value lines → no redaction.
    assert n == 0
    assert clean == prose


def test_empty_text_is_safe():
    assert redact_secrets("") == ("", 0)
    assert redact_secrets("   ") == ("   ", 0)


def test_figma_link_redacted_as_credential():
    clean, n = redact_secrets("design: https://www.figma.com/file/AbC123/Design-System")
    assert n == 1 and "figma.com/file/AbC123" not in clean


def test_pipeline_redacts_included_docs_before_chunking():
    """The ingest wiring (redact_included) must scrub secrets from EVERY included doc, so no
    chunk/payload/embedding ever sees a live credential — defense-in-depth past exclusion."""
    from pathlib import Path

    from rageval.ingest import redact_included
    from rageval.sources.base import SourceDoc

    docs = [
        SourceDoc(project_id="x", source_set="atlas",
                  doc_path=Path("settings.md"), doc_type="other", ext="md",
                  raw_text="Brand: Keep Me\napi_key: deadbeefdeadbeefdeadbeef99", folder_meta={}),
        SourceDoc(project_id="y", source_set="atlas",
                  doc_path=Path("overview.md"), doc_type="spec", ext="md",
                  raw_text="A clean product description with no secrets.", folder_meta={}),
    ]
    redacted, total, n_pii = redact_included(docs)
    assert total >= 1
    assert "Brand: Keep Me" in redacted[0].raw_text          # content preserved
    assert "deadbeefdeadbeefdeadbeef99" not in redacted[0].raw_text  # secret gone
    assert redacted[1].raw_text == docs[1].raw_text          # clean doc untouched


# --- PII redaction (emails) — a DISTINCT, POLICY-AWARE guardrail ----------------------

def test_personal_email_in_internal_doc_is_redacted():
    """A personal-looking email in an INTERNAL doc (e.g. a pitch/settings/spec) is PERSONAL PII
    → redacted (e.g. an author's personal address pasted into an internal pitch)."""
    clean, n = redact_pii("Pitch contact: alex.rivera@example.com for the partner deal.",
                          doc_type="spec")
    assert n == 1
    assert "alex.rivera@example.com" not in clean
    assert EMAIL_PLACEHOLDER in clean
    assert "Pitch contact:" in clean  # surrounding prose preserved


def test_published_support_email_in_description_is_kept():
    """A published business contact in a PUBLIC-FACING doc (description) is NOT PII → KEPT, so a
    legitimate query ('what's the support email?') still works."""
    clean, n = redact_pii("Questions? Email support@app.com any time.", doc_type="description")
    assert n == 0
    assert "support@app.com" in clean
    assert EMAIL_PLACEHOLDER not in clean


def test_role_based_email_is_kept_even_in_internal_doc():
    """A role/functional local-part (info@, support@, …) is a business address → kept even in an
    internal doc, regardless of doc_type."""
    clean, n = redact_pii("Internal note: route to info@company.com.", doc_type="spec")
    assert n == 0
    assert "info@company.com" in clean


def test_personal_email_in_public_doc_is_kept_published_contact():
    """Even a personal-looking address is treated as a published contact when it appears in a
    public-facing doc (the author chose to publish it there)."""
    clean, n = redact_pii("Reach the founder jane.doe@studio.com on our store page.",
                          doc_type="promo")
    assert n == 0
    assert "jane.doe@studio.com" in clean


def test_email_password_pair_still_hits_credential_path_not_pii():
    """An `email:password` blob is a CREDENTIAL — the secret path must catch it, and the PII
    pass must NOT also fire on the (already-scrubbed) address. The two categories stay
    distinct: this is the whole point of running redact_secrets before redact_pii."""
    clean, n_sec, n_pii = redact("login blob bot@example.com:Sup3rSecret! here", doc_type="spec")
    assert n_sec >= 1                       # credential path fired
    assert "bot@example.com:Sup3rSecret!" not in clean
    assert CRED_PLACEHOLDER in clean
    # The email was consumed as part of the credential, so the PII pass redacts 0 here.
    assert n_pii == 0
    assert EMAIL_PLACEHOLDER not in clean


def test_pii_does_not_touch_non_pii_text():
    """Prices, ids, versions, and @-mentions must NOT be mistaken for emails (high precision)."""
    text = "Price $1,200; ID 4155551234; version 1.2.3; ping @team in chat."
    clean, n = redact_pii(text, doc_type="spec")
    assert n == 0
    assert clean == text


def test_policy_can_disable_personal_email_redaction():
    """The human owns the policy: turning redact_personal_emails off keeps everything."""
    policy = PiiPolicy(redact_personal_emails=False)
    clean, n = redact_pii("personal jane.doe@example.com here", doc_type="spec", policy=policy)
    assert n == 0 and "jane.doe@example.com" in clean


def test_combined_redact_scrubs_secret_and_personal_email_in_internal_doc():
    """redact() returns (clean, n_secrets, n_pii): an INTERNAL settings doc with both a key AND
    a personal email gets each scrubbed and counted in its own category."""
    text = (
        "Brand: Tide Tracker\n"
        "api_key: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6\n"
        "owner: jane.doe@example.com\n"
    )
    clean, n_sec, n_pii = redact(text, doc_type="metadata")  # settings.md → metadata (internal)
    assert n_sec >= 1 and n_pii == 1
    assert "Brand: Tide Tracker" in clean                 # content preserved
    assert "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6" not in clean  # secret gone
    assert "jane.doe@example.com" not in clean             # personal PII gone
    assert KEY_PLACEHOLDER in clean and EMAIL_PLACEHOLDER in clean


def test_pii_empty_text_is_safe():
    assert redact_pii("", doc_type="spec") == ("", 0)
    assert redact_pii("   ", doc_type="spec") == ("   ", 0)
