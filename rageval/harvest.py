"""Deterministic per-project metadata harvest from `back/config.yaml` (structured extraction).

WHY this module exists (and why it's NOT an adapter or an embedded doc):
  Each project ships a `back/config.yaml` that holds the most AUTHORITATIVE structured
  facts about the app — its name, its website domain, its store number/bundle id.
  But `.yaml` is deliberately NOT in `corpus-rules.allow_ext`: config.yaml is metadata,
  not narrative, AND it interleaves clean fields with SECRET blocks (third-party
  analytics/integration secrets — api keys, tokens, session ids). Embedding it would both
  dilute top-k AND risk leaking secrets.

  So config.yaml is never discovered as a SourceDoc and never embedded. Instead this is a
  separate, DETERMINISTIC harvest step (parse the YAML, lift a WHITELIST of clean fields)
  that feeds the metadata sidecar — exactly the "structured data → SQL facet, not vector
  search" split the sidecar exists for. A query like "what are the website URLs for
  fruit-themed apps?" then answers from BOTH sources: theme from enrich (semantic), and
  domain/landing_url from this harvest (structured).

SECURITY — field WHITELIST, never a blacklist:
  We read ONLY `app.name`, `app.domain`, `app.number`, `app.bundle_id`, `app.localization`.
  The third-party analytics/integration secret blocks (and any `*_key`/`*_token`/
  secret `*_id`) are NEVER read or stored. A whitelist (not a blacklist) means a NEW secret
  field added upstream is excluded BY DEFAULT — fail-closed, the only safe default for
  credential-adjacent data. As a belt-and-braces guard, `_clean_str` also refuses any value
  whose key looks secret, so the whitelist can't be defeated by a renamed field.

CONTACT EMAIL — best-effort, honest, BOUNDED (we do NOT render the PHP templates):
  The public PHP pages build a support address as `<localpart>@<?= ...DOMAIN... ?>`. We do a
  tiny, conservative scan: ONLY when a template literally writes `<localpart>@` immediately
  before a PHP token that clearly echoes the domain do we construct `<localpart>@<domain>`
  and store it FLAGGED `derived=true`. Anything more indirected is left UNRESOLVED (null).
  We never execute PHP and never guess — an unresolved email is an acceptable, honest outcome.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# The ONLY fields we lift from app.* — a WHITELIST. Anything not here (every secret block) is
# excluded by construction. Maps the YAML key under `app:` → the ProjectRecord field name.
_APP_FIELD_WHITELIST: dict[str, str] = {
    "name": "app_name",
    "domain": "domain",
    "number": "app_number",
    "bundle_id": "bundle_id",
    "localization": "localization",
}

# Belt-and-braces: even within the whitelist, refuse a value whose KEY looks like a secret.
# (Defense-in-depth — the whitelist already excludes these, but a renamed upstream field that
# accidentally collided with a whitelisted name must still never be stored.)
_SECRET_KEY_RE = re.compile(r"(_key|_token|secret|password|api_key|session_id)", re.IGNORECASE)


@dataclass
class ConfigHarvest:
    """The clean, non-secret fields lifted from one project's back/config.yaml.

    Every field is Optional: config.yaml is absent in a small fraction of projects, and any individual
    field may be missing. `present` records whether config.yaml existed at all (drives the
    coverage report's "no config.yaml" flag). NO secret ever lands in here."""
    present: bool = False
    app_name: str | None = None
    domain: str | None = None
    app_number: str | None = None
    bundle_id: str | None = None
    localization: str | None = None
    contact_emails: list[str] | None = None  # derived from PHP templates (FLAGGED derived)
    contact_emails_derived: bool = False      # True iff contact_emails was constructed, not literal

    @property
    def landing_url(self) -> str | None:
        """Derived: the website is https://<domain>. None when no domain was harvested."""
        if not self.domain:
            return None
        return f"https://{self.domain}"


def _clean_str(key: str, value: object) -> str | None:
    """Coerce a whitelisted YAML scalar to a clean string, or None.

    Refuses non-scalars, empties, redaction placeholders, and — as a guard — any value whose
    KEY matches the secret pattern (so the whitelist cannot be defeated by a renamed field)."""
    if value is None or isinstance(value, (dict, list)):
        return None
    if _SECRET_KEY_RE.search(key):
        return None
    s = str(value).strip().strip("'\"").strip()
    if not s or s.startswith("[REDACTED"):
        return None
    return s


def parse_config_yaml(text: str) -> dict[str, str]:
    """Parse config.yaml text → ONLY the whitelisted clean app.* fields (a dict).

    Pure + deterministic (no I/O) so it's trivially unit-testable. SECRET blocks are never
    touched: we read `app:` and pull only the whitelisted keys; the third-party
    analytics/integration secret blocks (and anything else) are ignored entirely. Returns {}
    on a malformed file."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    app = data.get("app")
    if not isinstance(app, dict):
        return {}
    out: dict[str, str] = {}
    for yaml_key, field_name in _APP_FIELD_WHITELIST.items():
        cleaned = _clean_str(yaml_key, app.get(yaml_key))
        if cleaned is not None:
            out[field_name] = cleaned
    return out


# --- contact email: a conservative, BOUNDED scan of the public PHP templates ----------------
# We only resolve the SIMPLE case: a literal local-part immediately followed by '@' and then a
# PHP echo of the domain, e.g.   support@<?= $config['app']['domain'] ?>   or   support@<?php echo $domain; ?>
# The local-part must be a plain mailbox token (letters/digits/._%+- ), and the PHP token that
# follows must reference a "domain" variable/key — otherwise we DO NOT resolve (leave null).
_EMAIL_TEMPLATE_RE = re.compile(
    r"([A-Za-z0-9._%+\-]+)@"               # a literal local-part, then '@'
    r"<\?(?:php\s+echo|=)\s*"               # opening PHP echo: <?php echo  OR  <?=
    r"[^?]*?domain[^?]*?\?>",               # ...something referencing 'domain'..., then ?>
    re.IGNORECASE,
)
_EMAIL_TEMPLATE_FILES = ("contact.php", "privacy.php", "index.php")


def derive_contact_emails(template_text: str, domain: str | None) -> list[str]:
    """Best-effort: construct `<localpart>@<domain>` ONLY for the simple literal-local-part case.

    Returns a (de-duplicated, ordered) list of constructed addresses, or [] when nothing simple
    matched or no domain is known. We NEVER fabricate a domain — without a harvested domain there
    is nothing to construct, so we return []. Complex/indirected templates resolve to nothing
    here (honest UNRESOLVED) rather than a guess."""
    if not domain or not template_text:
        return []
    seen: list[str] = []
    for m in _EMAIL_TEMPLATE_RE.finditer(template_text):
        local = m.group(1).strip(".")
        if not local:
            continue
        addr = f"{local}@{domain}"
        if addr not in seen:
            seen.append(addr)
    return seen


def harvest_project(project_dir: Path) -> ConfigHarvest:
    """Read `<project_dir>/back/config.yaml` (if present) → a ConfigHarvest of CLEAN fields.

    Degrades gracefully: a missing config.yaml yields `present=False` and all-None fields (the
    coverage report flags those). Then a bounded best-effort contact-email derivation from the
    public PHP templates. NO secret block is ever read; NO value is ever fabricated."""
    h = ConfigHarvest()
    back = project_dir / "back"
    cfg = back / "config.yaml"
    if cfg.is_file():
        try:
            fields = parse_config_yaml(cfg.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            fields = {}
        h.present = True
        h.app_name = fields.get("app_name")
        h.domain = fields.get("domain")
        h.app_number = fields.get("app_number")
        h.bundle_id = fields.get("bundle_id")
        h.localization = fields.get("localization")

    # Best-effort contact email from the public PHP templates (only if we have a domain).
    if h.domain and back.is_dir():
        for fname in _EMAIL_TEMPLATE_FILES:
            tmpl = back / fname
            if not tmpl.is_file():
                continue
            try:
                emails = derive_contact_emails(tmpl.read_text(encoding="utf-8", errors="replace"),
                                               h.domain)
            except OSError:
                emails = []
            if emails:
                h.contact_emails = emails
                h.contact_emails_derived = True  # constructed from <local>@<domain>, FLAGGED
                break
    return h
