"""Classification POLICY for the bundled SAMPLE corpus (owned by the sample adapters, #37).

Every filename/asset heuristic that used to live in the shared `sources/base.py` — which files
are derived store-listing duplicates, which are metadata-only, which docs/*.txt are content vs
config — is corpus-SPECIFIC knowledge, so it lives HERE with the adapters that own the sample
corpus (northwind/atlas), NOT in the engine core.

Two consumers use this policy:

  1. DISCOVERY (the adapters): the helper functions below assign a doc's coarse `doc_type` as it's
     discovered (store-listing detection drives the promo-fallback dedup; settings.md → 'metadata';
     docs/*.txt content-vs-config → the real doc_type). This is where the concrete filename lists
     are applied to shape what the adapter yields.

  2. CLASSIFICATION (`classify.py` via the adapter contract): `sample_classification_policy()`
     declares the per-corpus `allow_ext` (the content extensions this corpus contributes) plus
     any FileRules the classifier applies generically. The engine core provides the MECHANISM
     (a rule can drop / mark metadata-only / retype); this module supplies the POLICY.

A different corpus supplies its own policy (or none — the generic default). The engine core has
no knowledge of any filename below.
"""

from __future__ import annotations

import re

from .base import ClassificationPolicy

# --- store-listing detection (drives the promo-fallback dedup at discovery) --------------------
# App-store / Google-Play listing txt files are DERIVED from a project's canonical description
# (reformatted to each store's length/policy limits). They are near-duplicates that triplicate a
# project in top-k retrieval and hurt result diversity. The adapters treat them as a promo
# FALLBACK — yielded ONLY when the project has no canonical description.
#   description_app_store.txt · description_google_play.txt   (the canonical pair)
#   <anything>_app_store.txt  · <anything>_google_play.txt     (per-locale / variant)
#   store_listing.txt · app_store.txt · google_play.txt        (bare variants)
_STORE_LISTING_TXT = re.compile(
    r"(^|.*[_-])(app[_-]?store|google[_-]?play|play[_-]?store|store[_-]?listing)\.txt$",
    re.IGNORECASE,
)
# A file whose name marks it the CANONICAL description (preferred over any store listing).
_CANONICAL_DESCRIPTION = re.compile(r"^description\.(md|txt)$", re.IGNORECASE)

# --- docs/*.txt content-vs-config disambiguation ----------------------------------------------
# A few docs/*.txt filenames are genuine config/credential dumps, but several are CONTENT:
#   description.txt → store/app copy; ideas.txt / design.txt → gameplay / visual-theme concept.
# So we map by FILENAME instead of blanket-tagging the dir.
_CONFIG_TXT_NAMES = frozenset({"accounts.txt", "settings.txt", "setup.txt"})
_CONTENT_TXT_TYPES: dict[str, str] = {
    "description.txt": "description",  # store/app copy
    "ideas.txt": "spec",              # gameplay concept
    "design.txt": "spec",             # visual/theme requirements
}

# --- settings.md → metadata-only (enriched, NOT embedded) -------------------------------------
# settings.md is rich per-project METADATA (Brand/Theme/Mascot/Category). It's metadata, not
# narrative: embedding many such docs dilutes top-k with key:value boilerplate. Tagged 'metadata'
# so the indexer SKIPS it while enrich CONSUMES it as the preferred structured source.
_METADATA_FILENAMES = frozenset({"settings.md"})


def is_store_listing_txt(name: str) -> bool:
    """True if `name` is a derived app-store / Google-Play listing .txt file."""
    return bool(_STORE_LISTING_TXT.match(name))


def is_metadata_only_file(name: str) -> bool:
    """True if `name` is a metadata file routed to enrich only (e.g. settings.md): not embedded
    as retrieval chunks, but fed to the metadata-enrichment step."""
    return name.lower() in _METADATA_FILENAMES


def docs_txt_doc_type(name: str) -> str:
    """doc_type for a .txt file living directly under a `docs/` dir.

    Returns 'config' for genuine credential/setup dumps (excluded downstream), or the real
    content type ('description'/'spec') for content-named files. Anything else defaults to
    'config' (conservative: an unknown docs/*.txt is more likely a dump than product copy)."""
    low = name.lower()
    if low in _CONTENT_TXT_TYPES:
        return _CONTENT_TXT_TYPES[low]
    if low in _CONFIG_TXT_NAMES:
        return "config"
    return "config"


def is_canonical_description(name: str) -> bool:
    """True if `name` is the canonical product description (description.md / description.txt)."""
    return bool(_CANONICAL_DESCRIPTION.match(name))


def sample_classification_policy() -> ClassificationPolicy:
    """The sample corpus's declared classification policy (#37).

    The concrete filename doc_type shaping happens at DISCOVERY (the helpers above), so by the
    time classify.py runs, the sample corpus's doc_types + corpus-rules.yaml already express its
    policy. This declares the per-corpus content EXTENSIONS the sample contributes: php/html/htm
    are allowed ONLY because the adapters' index.php/html PROMO FALLBACK yields stripped, visible
    landing-page copy as a last-resort product source. (md/txt/docx are the baseline every corpus
    shares via corpus-rules.yaml; declaring them here too is harmless — the classifier unions.)

    No FileRules are declared here: the sample adapters already assign the right doc_type at
    discovery (store-listing → promo fallback, settings.md → 'metadata', docs/*.txt → config), and
    corpus-rules.yaml drops/keeps by doc_type. The FileRule mechanism exists for corpora that
    prefer to declare filename rules rather than shape doc_type in discovery (exercised by the
    synthetic second-corpus test)."""
    return ClassificationPolicy(
        allow_ext=frozenset({"md", "txt", "docx", "php", "html", "htm"}),
        file_rules=(),
    )


def atlas_classification_policy() -> ClassificationPolicy:
    """The atlas (multi-format, legacy-heavy) corpus's policy — the sample policy PLUS `pdf`.

    Only the atlas adapter reads born-digital PDFs (#39), so `pdf` is declared HERE, per-corpus,
    rather than in the shared baseline: adding a format for the multi-format corpus can never
    silently change how another corpus (northwind) classifies. The bundled sample data ships no
    PDFs, so this is inert for the sample and exercised by the PDF fixture tests + real corpora."""
    base = sample_classification_policy()
    return ClassificationPolicy(
        allow_ext=base.allow_ext | frozenset({"pdf"}),
        file_rules=base.file_rules,
    )
