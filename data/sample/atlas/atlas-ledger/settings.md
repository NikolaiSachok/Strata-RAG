# Project Settings

Brand: Lemon Ledger
Category: budgeting
Theme: citrus

# The secret keys below must be REDACTED at ingest, but the Brand line above must be KEPT (it
# is the reason we keep settings.md). settings.md is INTERNAL metadata (doc_type 'metadata'),
# so the OWNER's personal email is PERSONAL PII and is redacted too — while a published
# support@ contact in a public store description would be KEPT (policy-aware, not blanket).
owner: jordan.lee@example.com
# A person NAME in free text (no '@', no key:value shape) — the lightweight regex detector is
# structurally blind to it; Presidio's NER catches it as a PERSON entity (the comparison harness
# surfaces exactly this kind of regex-vs-NER disagreement). Fictional name.
internal note: escalate billing questions to Sarah Williams on the finance side.
api_key: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
sportsdata_io_key: 9f8e7d6c5b4a39281706f5e4d3c2b1a0
figma_link: https://www.figma.com/file/AbC123dEf456/Lemon-Ledger-Design
