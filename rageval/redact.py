"""Secret redaction — strip live credentials from document text BEFORE it is indexed.

WHY this exists (a real, critical finding): an enterprise corpus is full of files authored
by humans and build tools, and some of them embed LIVE SECRETS — an API key pasted into a
`docs/accounts.txt`, a login in `settings.txt`, a keystore password next to a `.jks`
reference. If those land in the vector index, the secret is now: (a) embedded into vectors,
(b) sitting in the chunk PAYLOAD as plaintext, and (c) retrievable — one well-phrased query
("what is the API key?") exfiltrates it. That is a credential leak with the RAG engine as
the delivery mechanism.

THE DEFENSE: redact secret VALUES at ingest time, after extraction but before chunking, for
EVERY included document — defense-in-depth alongside the exclusion rules (which drop whole
credential-dump files). Even a credential that survives in a file we chose to KEEP (e.g.
`settings.md`, kept because it carries the `Brand:` field) gets its key scrubbed here.

DESIGN PRINCIPLES
  * Redact the VALUE, preserve everything else. We never drop a whole line of prose; we
    replace just the secret token with a placeholder so the surrounding text (esp.
    `Brand:` lines and product copy) stays intact and useful for retrieval/enrichment.
  * Two complementary detectors, because secrets show up two ways:
      1. SHAPE   — high-entropy / structured tokens that are secrets by their form alone
                   (long hex, base64-ish blobs, UUID-style keys). Catches keys with no label.
      2. CONTEXT — `key: value` style lines whose KEY names a secret (api_key, password,
                   token, credential, login, ...). Catches short/low-entropy secrets a
                   shape detector would miss, by trusting the label.
  * Plus a few SERVICE TELLS the audit named explicitly (sportsdata.io keys, figma.com
    private links, email:password pairs, secrets adjacent to a `.jks` keystore reference).

Pure and deterministic → fully unit-testable, which matters: a redactor you can't test is a
redactor you can't trust. Returns (clean_text, n_redactions) so the count surfaces in the
dry-run manifest and the REPORT (visibility = auditability).

PII vs SECRETS — two DISTINCT guardrail categories (see redact_pii below). Secret redaction
strips live CREDENTIALS (keys, key:value pairs, email:password blobs). But a BARE email
address (`ns@example.com`) is not a credential — it's PERSONAL DATA (PII). In a regulated /
insurance setting, embedding PERSONAL PII into a retrievable index is its own compliance
problem (data minimization, GDPR), independent of credential leakage. So we run a SEPARATE,
complementary PII pass. The two passes compose: secret redaction runs FIRST (so an
`email:password` credential is caught as a credential, not mistaken for a bare PII email),
then the PII pass handles the remaining standalone addresses.

POLICY-AWARE, NOT BLANKET. A blanket "redact every email" rule is wrong: it would scrub
PUBLISHED business contacts and break a legitimate query ("what's the support email for app
X?"). Real data classification distinguishes PERSONAL PII from PUBLISHED business contacts.
So the PII pass is CONTEXT-AWARE — it takes the doc's provenance (doc_type) and a configurable
policy (PiiPolicy, sourced from corpus-rules.yaml; the human sets policy, the engine enforces —
same trust boundary as corpus_intent). An email is KEPT if EITHER it lives in a PUBLIC-FACING
doc (promo/description/store-listing/landing) OR its local-part is ROLE-BASED (support@, info@,
…). Otherwise — a personal-looking address in an internal doc (pitch/settings/spec/notes) — it
is redacted to `[REDACTED_EMAIL]` (e.g. an author's personal address pasted into an internal
pitch deck or settings file).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pii import (
    ENTITY_EMAIL,
    PLACEHOLDERS,
    PiiDetector,
    PiiSpan,
    get_pii_detector,
)

KEY_PLACEHOLDER = "[REDACTED_KEY]"
CRED_PLACEHOLDER = "[REDACTED_CREDENTIAL]"
EMAIL_PLACEHOLDER = "[REDACTED_EMAIL]"

# Words that, when they name the KEY in a `key: value` / `key = value` line, mean the VALUE
# is a secret to redact. Matched case-insensitively, allowing a space or underscore (api key,
# api_key, API-KEY).
_SECRET_KEY_WORDS = (
    r"api[_ -]?key", r"secret(?:[_ -]?key)?", r"access[_ -]?token", r"auth[_ -]?token",
    r"token", r"password", r"passwd", r"pwd", r"credential", r"client[_ -]?secret",
    r"private[_ -]?key", r"login",
)

# --- CONTEXTUAL: `<secret-word> [:=] <value>` → redact the value (keep the label). ---------
# We keep the key and the separator so the line still reads "api_key: [REDACTED_KEY]".
# An optional key-name PREFIX is allowed before the secret word so compound keys are caught:
# `keystore_password`, `release_keystore_password`, `client.secret` → still redacted. The
# prefix must end on a word-boundary char (_/-/.) so we don't match unrelated words.
_KEY_PREFIX = r"(?:[\w.-]*[_.-])?"
_CONTEXT_RE = re.compile(
    r"(?im)^(?P<prefix>\s*[\"']?" + _KEY_PREFIX + r"(?:" + "|".join(_SECRET_KEY_WORDS)
    + r")[\"']?\s*[:=]\s*)(?P<value>\S[^\n]*?)\s*$"
)

# --- SHAPE: high-entropy / structured tokens that are secrets by form alone. ---------------
# UUID-style key (8-4-4-4-12 hex), often used as an API key.
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                      r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
# A long run of hex (>=24 chars) — e.g. a hex API key or token.
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{24,}\b")
# A base64-ish / token-ish blob (>=32 chars of url-safe base64 alphabet). The character-class
# requirement (must contain at least one digit AND one letter) avoids redacting long ordinary
# words. We check that inside the replacement function.
_B64_RE = re.compile(r"\b[A-Za-z0-9+/_\-]{32,}={0,2}\b")

# --- SERVICE TELLS named by the audit. -----------------------------------------------------
# figma.com private/design links (often paste-dumped into docs/*.txt).
_FIGMA_RE = re.compile(r"https?://(?:www\.)?figma\.com/\S+")
# email:password pair (a credential blob shape), e.g. user@example.com:s3cretpass
_EMAIL_PWD_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+:\S{4,}")


def _looks_like_secret_blob(tok: str) -> bool:
    """A base64-ish token is only treated as a secret if it mixes letters AND digits — that
    rules out long lowercase words / hyphenated slugs while keeping real keys."""
    return bool(re.search(r"[A-Za-z]", tok)) and bool(re.search(r"[0-9]", tok))


def redact_secrets(text: str) -> tuple[str, int]:
    """Redact secret VALUES in `text`, preserving all other content.

    Returns (clean_text, n_redactions). Order matters: we run CONTEXTUAL redaction first
    (it understands `key: value` structure and keeps the label), then SHAPE/SERVICE
    detectors mop up unlabelled secrets. Counting is by number of substitutions made.
    """
    if not text:
        return text, 0

    count = 0

    # 1. SERVICE TELLS that are unambiguous credentials → CRED placeholder.
    text, n = _EMAIL_PWD_RE.subn(CRED_PLACEHOLDER, text)
    count += n
    text, n = _FIGMA_RE.subn(CRED_PLACEHOLDER, text)
    count += n

    # 2. CONTEXTUAL key:value — redact the value, keep "key: " so prose/structure survive.
    def _ctx_sub(m: re.Match) -> str:
        nonlocal count
        value = m.group("value")
        # Don't double-redact an already-redacted line, and don't redact an empty value.
        if not value or value.strip() in (KEY_PLACEHOLDER, CRED_PLACEHOLDER):
            return m.group(0)
        count += 1
        # "login" / "password" / "credential" read as credentials; key-ish words as keys.
        word = m.group("prefix").lower()
        placeholder = CRED_PLACEHOLDER if re.search(
            r"password|passwd|pwd|credential|login", word) else KEY_PLACEHOLDER
        return f"{m.group('prefix')}{placeholder}"

    text = _CONTEXT_RE.sub(_ctx_sub, text)

    # 3. SHAPE detectors for unlabelled secrets → KEY placeholder.
    text, n = _UUID_RE.subn(KEY_PLACEHOLDER, text)
    count += n
    text, n = _HEX_RE.subn(KEY_PLACEHOLDER, text)
    count += n

    def _b64_sub(m: re.Match) -> str:
        nonlocal count
        tok = m.group(0)
        if tok in (KEY_PLACEHOLDER, CRED_PLACEHOLDER) or not _looks_like_secret_blob(tok):
            return tok
        count += 1
        return KEY_PLACEHOLDER

    text = _B64_RE.sub(_b64_sub, text)

    return text, count


# ---------------------------------------------------------------------------
# PII redaction — a SEPARATE, POLICY-AWARE guardrail category.
# ---------------------------------------------------------------------------
# DETECTION is now a PLUGGABLE BACKEND (see pii.py): the lightweight regex detector (DEFAULT)
# or Microsoft Presidio (optional NER). This module owns the POLICY that sits ABOVE the
# detector — keep PUBLISHED / role-based contacts, redact PERSONAL data — which is unchanged
# whichever detector runs. The regex detector finds emails (intentionally narrow / high
# precision — see pii.py); Presidio additionally labels PERSON / PHONE_NUMBER / IBAN / ….

# PHONE NUMBERS in the REGEX path: deliberately NOT redacted. A precise phone detector that does not
# false-positive on prices, ids, and version numbers (e.g. "1.2.3", "$1,200", "ID 4155551234")
# needs locale-aware parsing we don't want to fake here. Emails are the concrete requirement;
# phone redaction is a documented FUTURE EXTENSION (add a libphonenumber-backed detector and a
# new placeholder, then surface its count the same way).

# DEFAULT policy values. The human OWNS these (set them in corpus-rules.yaml `pii_policy`); the
# engine ENFORCES them — the same untrusted-proposes/human-enforces trust boundary as
# corpus_intent. role local-parts = functional business addresses (kept). public doc_types =
# published surfaces whose contacts are deliberately public (kept).
_DEFAULT_ROLE_LOCAL_PARTS = frozenset({
    "support", "info", "contact", "help", "hello", "sales", "privacy", "legal",
    "admin", "team", "press", "partnerships", "no-reply", "noreply", "feedback",
})
_DEFAULT_PUBLIC_DOC_TYPES = frozenset({"promo", "description", "store_listing", "landing"})


@dataclass(frozen=True)
class PiiPolicy:
    """Configurable PII policy. The HUMAN sets this (corpus-rules.yaml `pii_policy`); the engine
    enforces it. Distinguishes PERSONAL PII (redact) from PUBLISHED business contacts (keep)."""
    redact_personal_emails: bool = True       # master switch for personal-email redaction
    preserve_published_contacts: bool = True  # keep emails in public-facing docs
    role_local_parts: frozenset[str] = field(default_factory=lambda: _DEFAULT_ROLE_LOCAL_PARTS)
    public_doc_types: frozenset[str] = field(default_factory=lambda: _DEFAULT_PUBLIC_DOC_TYPES)

    @classmethod
    def from_rules(cls, data: dict | None) -> "PiiPolicy":
        """Build a policy from a corpus-rules.yaml `pii_policy:` block (None → defaults)."""
        data = data or {}
        role = data.get("role_local_parts")
        public = data.get("public_doc_types")
        return cls(
            redact_personal_emails=bool(data.get("redact_personal_emails", True)),
            preserve_published_contacts=bool(data.get("preserve_published_contacts", True)),
            role_local_parts=frozenset(str(x).lower() for x in role) if role is not None
            else _DEFAULT_ROLE_LOCAL_PARTS,
            public_doc_types=frozenset(str(x).lower() for x in public) if public is not None
            else _DEFAULT_PUBLIC_DOC_TYPES,
        )


DEFAULT_PII_POLICY = PiiPolicy()


def _keep_span(span: PiiSpan, doc_type: str, policy: PiiPolicy) -> bool:
    """POLICY: True if this detected PII span should be KEPT (not redacted).

    This is the keep-or-redact decision — backend-AGNOSTIC, sits ABOVE the detector, and is
    UNCHANGED whether the span came from the regex or the Presidio detector.

      * EMAIL_ADDRESS — KEPT iff it's a PUBLISHED business contact: EITHER it lives in a
        public-facing doc OR its local-part is a role/functional address (support@, info@, …).
      * Everything else (PERSON, PHONE_NUMBER, IBAN, CREDIT_CARD) is PERSONAL data → REDACT,
        UNLESS it appears in a public-facing doc and the policy preserves published surfaces
        (a founder's name/phone deliberately published on a store page is a published contact).
    """
    dt = (doc_type or "").lower()
    in_public_doc = policy.preserve_published_contacts and dt in policy.public_doc_types
    if span.entity_type == ENTITY_EMAIL:
        if in_public_doc:
            return True
        if (span.local_part or "").lower() in policy.role_local_parts:
            return True
        return False
    # Non-email personal entities: kept only when published on a public-facing surface.
    return in_public_doc


def redact_pii(text: str, *, doc_type: str = "", policy: PiiPolicy = DEFAULT_PII_POLICY,
               detector: PiiDetector | None = None) -> tuple[str, int]:
    """Redact PERSONAL PII in `text`, PRESERVING published business contacts.

    DETECTION is pluggable (`detector`, default = the configured backend via get_pii_detector;
    regex unless RAGEVAL_PII_BACKEND=presidio). The keep/redact POLICY here is UNCHANGED across
    backends: `doc_type` provenance + the configurable `policy` decide each detected span.

    Returns (clean_text, n_redactions) — the SAME shape as redact_secrets, so PII counts surface
    in the manifest exactly like secret counts. An `email:password` credential is left to
    redact_secrets (the regex backend skips it via a lookahead; Presidio runs after secrets too).
    """
    if not text or not policy.redact_personal_emails:
        return text, 0

    det = detector if detector is not None else get_pii_detector()
    # Feed the role local-parts as the detector's allow_list (a cheap pre-filter); the policy
    # below re-checks every returned span, so it stays the single source of truth.
    spans = det.detect(text, allow_list=policy.role_local_parts)

    # Decide keep/redact per span, then substitute right-to-left so earlier offsets stay valid.
    to_redact = [s for s in sorted(spans, key=lambda s: s.start, reverse=True)
                 if not _keep_span(s, doc_type, policy)]
    n = 0
    for span in to_redact:
        placeholder = PLACEHOLDERS.get(span.entity_type, EMAIL_PLACEHOLDER)
        text = text[:span.start] + placeholder + text[span.end:]
        n += 1
    return text, n


def redact(text: str, *, doc_type: str = "", pii_policy: PiiPolicy = DEFAULT_PII_POLICY,
           detector: PiiDetector | None = None) -> tuple[str, int, int]:
    """Run BOTH guardrail passes in the correct order and report each count separately.

    Order matters: secret redaction (context-free) runs FIRST so an `email:password`
    credential is scrubbed as a credential; the policy-aware PII pass then redacts remaining
    PERSONAL data while preserving published contacts. `detector` (default = configured backend)
    selects the PII DETECTOR (regex | presidio); the secret pass and the keep/redact policy are
    unchanged. Returns (clean_text, n_secrets, n_pii) so callers surface the two distinct
    guardrails independently in the manifest."""
    clean, n_secrets = redact_secrets(text)
    clean, n_pii = redact_pii(clean, doc_type=doc_type, policy=pii_policy, detector=detector)
    return clean, n_secrets, n_pii
