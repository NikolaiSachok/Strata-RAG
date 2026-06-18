"""The `SourceDoc` record and the abstract `SourceAdapter` interface.

These two types are the contract between "where documents come from" (corpus-specific,
behind an adapter) and "what we do with them" (corpus-agnostic: classify, chunk, embed,
index). Keep this module tiny and dependency-light — it's the shape everything else
agrees on.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# --- provenance-aware dedup: store-listing detection (shared by every adapter) ----------
# App-store / Google-Play listing txt files are DERIVED from a project's canonical
# description (reformatted to each store's length/policy limits). They are near-duplicates
# that triplicate a project in top-k retrieval and hurt result diversity. Adapters treat
# them as a promo FALLBACK — yielded ONLY when the project has no canonical description —
# exactly like the index.* landing-page fallback. One regex here = one source of truth.
#
# Recognised store-listing names (case-insensitive, .txt only):
#   description_app_store.txt · description_google_play.txt   (the canonical pair)
#   <anything>_app_store.txt  · <anything>_google_play.txt     (per-locale / variant)
#   store_listing.txt · app_store.txt · google_play.txt        (bare variants)
_STORE_LISTING_TXT = re.compile(
    r"(^|.*[_-])(app[_-]?store|google[_-]?play|play[_-]?store|store[_-]?listing)\.txt$",
    re.IGNORECASE,
)
# A file whose name marks it the CANONICAL description (preferred over any store listing).
_CANONICAL_DESCRIPTION = re.compile(r"^description\.(md|txt)$", re.IGNORECASE)


def is_store_listing_txt(name: str) -> bool:
    """True if `name` is a derived app-store / Google-Play listing .txt file."""
    return bool(_STORE_LISTING_TXT.match(name))


# --- docs/*.txt content-vs-config disambiguation (shared by every adapter) --------------
# CORRECTION to an earlier over-broad rule: the adapters used to tag EVERY .txt directly under
# docs/ as doc_type 'config' (which corpus-rules drops). That produced FALSE NEGATIVES — real
# product content named *.txt was silently dropped. The truth (from a layout survey): a few
# docs/*.txt filenames are genuine config/credential dumps, but several are CONTENT:
#   description.txt → store/app copy (a 'description'); ideas.txt / design.txt → gameplay /
#   visual-theme concept (a 'spec'). So we map by FILENAME instead of blanket-tagging the dir.
# Genuine config that must STAY excluded (credentials / Figma-link / build setup dumps).
_CONFIG_TXT_NAMES = frozenset({"accounts.txt", "settings.txt", "setup.txt"})
# Content-named docs/*.txt → the doc_type they really are (so classify.py KEEPS them).
_CONTENT_TXT_TYPES: dict[str, str] = {
    "description.txt": "description",  # store/app copy
    "ideas.txt": "spec",              # gameplay concept
    "design.txt": "spec",             # visual/theme requirements
}


# --- settings.md → metadata-only (enriched, NOT embedded) -------------------------------
# settings.md is rich per-project METADATA (Brand/Theme/Mascot/Category) for some projects,
# thin/absent for others. It is metadata, not narrative: embedding many such docs dilutes
# top-k with key:value boilerplate. Decision: tag it doc_type 'metadata' so the indexer SKIPS
# it (excluded from the vector index) while the enrich step still CONSUMES it as the preferred
# structured source. A 'metadata' doc is INCLUDED-but-metadata_only (see classify.py).
_METADATA_FILENAMES = frozenset({"settings.md"})


def is_metadata_only_file(name: str) -> bool:
    """True if `name` is a metadata file routed to enrich only (e.g. settings.md): not
    embedded as retrieval chunks, but fed to the metadata-enrichment step."""
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


@dataclass(frozen=True)
class SourceDoc:
    """One document discovered in the corpus, normalised to a single shape.

    Every adapter, no matter how weird its corpus layout, yields this. Downstream code
    depends ONLY on these fields — never on the filesystem.

    Fields:
      project_id   — stable id of the project this doc belongs to (e.g. "0007" or
                     "atlas-ledger"). Aggregation/grouping is by this.
      source_set   — which adapter/family produced it (e.g. "northwind", "atlas").
                     Lets you ask "themes used in BOTH source-sets" (set intersection).
      doc_path     — absolute path on disk (used for citations + the chunk inspector;
                     NEVER embedded, so a real path is never committed).
      doc_type     — coarse kind: "description" | "readme" | "promo" | "spec" |
                     "changelog" | "plan" | "agent_doc" | "other". The relevance
                     classifier reasons partly off this.
      ext          — file extension without the dot, lowercased ("md", "txt", "docx").
      raw_text     — the extracted plain text (this IS what gets chunked + embedded).
      folder_meta  — anything the adapter could glean from the FOLDER name/structure
                     (e.g. a theme or brand hint encoded in the directory). A dict so
                     adapters can pass through whatever they know cheaply.
    """

    project_id: str
    source_set: str
    doc_path: Path
    doc_type: str
    ext: str
    raw_text: str
    folder_meta: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """A stable, globally-unique id for this document (across source-sets)."""
        return f"{self.source_set}/{self.project_id}/{self.doc_path.name}"


class SourceAdapter(abc.ABC):
    """Abstract base every concrete adapter implements.

    A concrete adapter is constructed with the *root* of its corpus family and knows
    two things: its `source_set` name, and how to `discover()` SourceDocs under that
    root. That's the entire surface area. To support a new corpus you subclass this and
    register it via the public `registry.register_adapter(folder, cls)` API — no change
    anywhere else in the engine, and no edit to the registry's core mapping.
    """

    #: short, stable identifier for this family of projects.
    source_set: str = "base"

    def __init__(self, root: Path):
        self.root = Path(root)

    @abc.abstractmethod
    def discover(self) -> Iterable[SourceDoc]:
        """Walk the corpus root and yield one SourceDoc per candidate document.

        IMPORTANT: an adapter yields *candidates*, not the final include list. It does
        NOT decide relevance — that's classify.py's job, against corpus-rules.yaml. The
        separation matters: discovery is "what files exist and how do I read them";
        classification is "which of those are signal for this corpus_intent". Keeping
        them apart is what makes the dry-run manifest able to show EXCLUDED files (the
        adapter found them; a rule dropped them).
        """
        raise NotImplementedError

    # ---- small shared helpers concrete adapters can reuse ------------------

    @staticmethod
    def read_text(path: Path) -> str:
        """Read a text-like file (.md/.txt) tolerantly."""
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def read_docx(path: Path) -> str:
        """Extract paragraph text from a .docx.

        WHY this lives here: legacy corpora hide real content in Word documents. The
        whole "document intelligence over legacy docs" value proposition hinges on being
        able to parse them. python-docx walks the document's paragraph runs; we join
        them with newlines. Tables/headers/footers are out of scope for the demo but
        would extend here.
        """
        from docx import Document  # lazy import: tests that don't parse .docx don't pay for it

        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(parts)
