"""Structured-fact harvest for the bundled SAMPLE corpus (owned by the sample adapters).

This is the corpus-SPECIFIC half of the Phase-4 structured-harvest decouple (#36). Everything
here — WHICH descriptor file a project ships, WHICH fields to lift from it, and the best-effort
contact-email derivation — is knowledge about the sample corpus's layout, so it lives with the
adapters that own that corpus (northwind/atlas), NOT in the engine core. The core only defines
the `StructuredFact` shape + the reusable `FieldWhitelistHarvester` primitive (rageval.facts) and
consumes whatever facts an adapter emits; it knows none of the concrete field names below.

The engine core has NO knowledge of `back/config.yaml`, `app.name`, `app.domain`, etc. — a second
corpus with a totally different descriptor schema supplies its own harvester with no core change.

WHY these facts are NOT SourceDocs (never embedded): the descriptor is metadata, not narrative,
AND it interleaves clean fields with SECRET blocks (third-party analytics/integration secrets).
Embedding it would both dilute top-k AND risk leaking secrets. So it feeds the sidecar via the
structured-fact hook instead — exactly the "structured data → SQL facet, not vector search" split.

SECURITY — field WHITELIST, never a blacklist (the invariant, enforced by rageval.facts):
  We read ONLY the whitelisted `app.*` keys. The third-party analytics/integration secret blocks
  (and any `*_key`/`*_token`/secret `*_id`) are NEVER read or stored. A whitelist (not a blacklist)
  means a NEW secret field added upstream is excluded BY DEFAULT — fail-closed. The primitive also
  refuses any value whose key looks secret, so the whitelist can't be defeated by a renamed field.

CONTACT EMAIL — best-effort, honest, BOUNDED (we do NOT render the PHP templates):
  The public PHP pages build a support address as `<localpart>@<?= ...DOMAIN... ?>`. We do a tiny,
  conservative scan: ONLY when a template literally writes `<localpart>@` immediately before a PHP
  token that clearly echoes the domain do we construct `<localpart>@<domain>` and emit it with
  provenance "derived". Anything more indirected is left UNRESOLVED. We never execute PHP.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import yaml

from ..facts import FieldWhitelistHarvester, StructuredFact

# The ONLY keys we lift from the descriptor's `app.*` block — a WHITELIST. Maps the YAML key
# under `app:` → the sidecar FIELD name. Anything not here (every secret block) is excluded by
# construction. This concrete list is corpus-specific → it lives HERE, not in the engine core.
_APP_FIELD_WHITELIST: dict[str, str] = {
    "name": "app_name",
    "domain": "domain",
    "number": "app_number",
    "bundle_id": "bundle_id",
    "localization": "localization",
}

_HARVESTER = FieldWhitelistHarvester(_APP_FIELD_WHITELIST)

# The descriptor file this corpus ships (relative to the project dir).
_DESCRIPTOR_REL = ("back", "config.yaml")

# --- contact email: a conservative, BOUNDED scan of the public PHP templates ----------------
# Only the SIMPLE case: a literal local-part immediately followed by '@' and then a PHP echo of
# the domain, e.g.  support@<?= $config['app']['domain'] ?>  or  support@<?php echo $domain; ?>
_EMAIL_TEMPLATE_RE = re.compile(
    r"([A-Za-z0-9._%+\-]+)@"               # a literal local-part, then '@'
    r"<\?(?:php\s+echo|=)\s*"               # opening PHP echo: <?php echo  OR  <?=
    r"[^?]*?domain[^?]*?\?>",               # ...something referencing 'domain'..., then ?>
    re.IGNORECASE,
)
_EMAIL_TEMPLATE_FILES = ("contact.php", "privacy.php", "index.php")


def parse_config_yaml(text: str) -> dict[str, str]:
    """Parse descriptor text → ONLY the whitelisted clean app.* fields (a dict).

    Pure + deterministic (no I/O) so it's trivially unit-testable. SECRET blocks are never
    touched: we read `app:` and lift only the whitelisted keys via the core primitive (which also
    refuses secret-looking keys). Returns {} on a malformed file / no `app:` block."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    app = data.get("app")
    if not isinstance(app, dict):
        return {}
    return _HARVESTER.lift(app)


def derive_contact_emails(template_text: str, domain: str | None) -> list[str]:
    """Best-effort: construct `<localpart>@<domain>` ONLY for the simple literal-local-part case.

    Returns a (de-duplicated, ordered) list of constructed addresses, or [] when nothing simple
    matched or no domain is known. We NEVER fabricate a domain — without a harvested domain there
    is nothing to construct, so we return []. Complex/indirected templates resolve to nothing
    (honest UNRESOLVED) rather than a guess."""
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


def harvest_facts(entity_id: str, project_dir: Path) -> Iterator[StructuredFact]:
    """Yield StructuredFacts for one sample-corpus project from its `back/config.yaml` descriptor.

    Emits the whitelisted clean app.* fields (provenance "descriptor" — authoritative), a DERIVED
    `landing_url` (https://<domain>), and a best-effort DERIVED `contact_emails` list from the
    public PHP templates. Degrades gracefully: a missing descriptor yields nothing. NO secret is
    ever read; NO value is ever fabricated. This is the concrete policy for the sample corpus; a
    different corpus supplies its own `harvest_facts` — the engine core is unchanged."""
    back = project_dir / _DESCRIPTOR_REL[0]
    cfg = project_dir / Path(*_DESCRIPTOR_REL)
    domain: str | None = None
    if cfg.is_file():
        try:
            fields = parse_config_yaml(cfg.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            fields = {}
        for field_name, value in fields.items():
            yield StructuredFact(entity_id, field_name, value, provenance="descriptor")
        domain = fields.get("domain")
        # DERIVED: the website is https://<domain>. Flagged provenance so a consumer knows it was
        # constructed, not a literal descriptor field.
        if domain:
            yield StructuredFact(entity_id, "landing_url", f"https://{domain}",
                                 provenance="derived")

    # Best-effort contact email from the public PHP templates (only if we have a domain).
    if domain and back.is_dir():
        for fname in _EMAIL_TEMPLATE_FILES:
            tmpl = back / fname
            if not tmpl.is_file():
                continue
            try:
                emails = derive_contact_emails(
                    tmpl.read_text(encoding="utf-8", errors="replace"), domain)
            except OSError:
                emails = []
            if emails:
                yield StructuredFact(entity_id, "contact_emails", emails, provenance="derived")
                break


def harvest_facts_for(entity_id: str, project_dir: Path | None) -> list[StructuredFact]:
    """Convenience wrapper: collect harvest_facts into a list, tolerating a None project_dir
    (a synthetic/marker project with no on-disk root → no descriptor → no facts)."""
    if project_dir is None:
        return []
    return list(harvest_facts(entity_id, project_dir))
