"""AtlasAdapter — the "messy, multi-format, .docx-heavy" corpus shape.

This adapter models a harder corpus family: named-slug project folders, content split
across `docs/` (including legacy **.docx** files), plus `README.md` / `task.md` /
`prd.md`, and assorted `back/` and `back-<id>/` engineering folders. This is the
heterogeneous, legacy-shaped landscape that separates real document-intelligence work
from a toy: the signal is buried in Word files next to template noise.

KEY teaching point: the SAME `SourceDoc` shape comes out the other side. The pipeline
downstream cannot tell a Northwind project from an Atlas project — which is the whole
point of the adapter seam. (Models a real multi-format corpus layout, genericised.)

LAYOUT-AUDIT additions (driven by a survey of real project layouts):
  * PATH-AWARE .txt typing: `docs/*.txt` is consistently a config/credential/Figma dump →
    tagged 'config' (which corpus-rules drops); `promo/*.txt` is real store copy → 'promo'.
  * index.php / index.html PROMO FALLBACK: for the ~handful of layouts whose ONLY product
    source is a back/ landing page, we extract the page's visible promo copy — but ONLY
    when the project has no proper description doc, to avoid redundant duplicate copy.
  * STORE-LISTING PROMO FALLBACK (provenance-aware dedup): app-store / Google-Play listing
    .txt files (store_listing.txt, *_app_store.txt, *_google_play.txt) are DERIVED from the
    canonical description (reformatted to store limits) — near-duplicates that hurt top-k
    diversity. We yield them ONLY when the project has no canonical description, exactly like
    the index.* fallback. (Adapt the pipeline to the data; never edit the source.)
"""

from __future__ import annotations

import html as _html
import re
from typing import Iterable

from pathlib import Path

from .base import ClassificationPolicy, SourceAdapter, SourceDoc
from .sample_facts import harvest_facts_for, sample_declared_facets
from .sample_policy import (
    atlas_classification_policy,
    docs_txt_doc_type,
    is_metadata_only_file,
    is_store_listing_txt,
)

_TEXT_EXTS = {".md", ".txt"}
_DOCX_EXT = ".docx"
_PDF_EXT = ".pdf"
_INDEX_EXTS = {".php", ".html", ".htm"}

# A doc counts as a real "description" (which suppresses the index.php fallback) if its
# doc_type is one of these. Kept here so the fallback rule reads clearly.
_DESCRIPTION_TYPES = {"description", "promo"}


def _doc_type_for(rel_parts: tuple[str, ...], name: str) -> str:
    low = name.lower()
    ext = low.rsplit(".", 1)[-1] if "." in low else ""
    # settings.md = structured METADATA → 'metadata' (enriched, not embedded). Checked first.
    if is_metadata_only_file(low):
        return "metadata"
    if low in ("changelog.md", "changelog.txt"):
        return "changelog"
    if "implementation_plan" in low or low in ("task.md", "plan.md"):
        return "plan"
    if low in ("claude.md",):
        return "agent_doc"
    # promo/*.txt is real STORE COPY → keep as promo.
    if "promo" in rel_parts and (ext == "txt" or low.startswith("description")):
        return "promo"
    # docs/*.txt: mostly config/credential dumps, but content-named files (description/ideas/
    # design .txt) are real content → disambiguate by filename (shared rule in base.py).
    if "docs" in rel_parts and ext == "txt":
        return docs_txt_doc_type(low)
    if low in ("prd.md", "spec.md") or low.startswith("description"):
        # prd/spec are excluded by filename rules downstream; description is real content.
        return "spec" if low.startswith(("prd", "spec")) else "description"
    if low == "readme.md":
        return "readme"
    if "docs" in rel_parts:
        return "spec"
    return "other"


def strip_markup(raw: str) -> str:
    """Reduce an index.php / index.html page to its VISIBLE text.

    A landing page's promo copy is the in-scope product content; the markup, scripts, and
    server code are not. We drop <script>/<style> blocks and PHP `<?php ... ?>` islands,
    strip the remaining tags, unescape HTML entities, and collapse whitespace. Deliberately
    small and dependency-free (no bs4) — enough to recover headline/paragraph copy, which is
    all the fallback needs, and trivially testable.
    """
    if not raw:
        return ""
    text = re.sub(r"<\?php.*?\?>", " ", raw, flags=re.DOTALL | re.IGNORECASE)  # PHP islands
    text = re.sub(r"<\?.*?\?>", " ", text, flags=re.DOTALL)                    # short-echo tags
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)                   # comments
    text = re.sub(r"<[^>]+>", " ", text)                                       # remaining tags
    text = _html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()                                   # collapse ws
    return text


class AtlasAdapter(SourceAdapter):
    source_set = "atlas"

    def discover(self) -> Iterable[SourceDoc]:
        if not self.root.exists():
            return
        for project_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            project_id = project_dir.name
            # Two passes per project so the index.php fallback can know whether a real
            # description already exists (don't duplicate copy when one does).
            docs: list[SourceDoc] = []
            index_candidates: list = []  # (path, raw_text) landing-page fallbacks
            store_listing_candidates: list = []  # (path, rel_parts, text) derived store copy

            for path in sorted(project_dir.rglob("*")):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                rel_parts = path.relative_to(project_dir).parts[:-1]

                # --- .docx: the legacy document path ---------------------------
                if ext == _DOCX_EXT:
                    try:
                        text = self.read_docx(path)
                    except Exception:  # noqa: BLE001 — a corrupt .docx shouldn't crash discovery
                        continue
                    doc_type = "description" if path.stem.lower().startswith("description") else "spec"
                    docs.append(self._mk(project_id, path, doc_type, "docx", text, project_dir))
                    continue

                # --- .pdf: born-digital PDF text (#39) -------------------------
                # A PDF yields a SourceDoc like any other document; the extractor carries page
                # provenance in the text. Detection (MAJOR-2): a FULLY text-less (scanned) PDF is
                # flagged `pdf_scanned` (manifest coverage warning, needs OCR) — never a silent blind
                # spot. A PARTIAL doc (some text pages + some image pages) KEEPS its extracted text
                # and only flags the image pages for OCR; its real text is embedded normally.
                if ext == _PDF_EXT:
                    try:
                        extraction = self.read_pdf(path)
                    except Exception:  # noqa: BLE001 — a corrupt PDF shouldn't crash discovery
                        continue
                    doc_type = ("description" if path.stem.lower().startswith("description")
                                else "spec")
                    meta = {"project_dir": project_dir.name}
                    if extraction.scanned:
                        # No usable text layer at all → mark it; discovery still yields the doc, the
                        # manifest flags it, and no empty chunk is embedded (there is no real text).
                        meta["pdf_scanned"] = True
                    elif extraction.needs_ocr_pages:
                        # Partial: real text extracted; SOME pages need OCR — flag WITHOUT dropping.
                        meta["pdf_ocr_pages"] = list(extraction.needs_ocr_pages)
                    docs.append(SourceDoc(
                        project_id=project_id, source_set=self.source_set, doc_path=path,
                        doc_type=doc_type, ext="pdf",
                        raw_text=extraction.text(), folder_meta=meta))
                    continue

                # --- plain text candidates -------------------------------------
                if ext in _TEXT_EXTS:
                    try:
                        text = self.read_text(path)
                    except OSError:
                        continue
                    # Defer DERIVED store-listing txt; it's a fallback, not unconditional copy.
                    if ext == ".txt" and is_store_listing_txt(path.name):
                        store_listing_candidates.append((path, rel_parts, text))
                        continue
                    docs.append(self._mk(project_id, path,
                                         _doc_type_for(rel_parts, path.name),
                                         ext.lstrip("."), text, project_dir))
                    continue

                # --- index.php / index.html: collect as a POSSIBLE fallback ----
                # Only back/index.* and a ROOT index.* qualify (not nested asset html).
                if ext in _INDEX_EXTS and path.stem.lower() == "index":
                    in_back = bool(rel_parts) and rel_parts[0].lower().startswith("back")
                    at_root = len(rel_parts) == 0
                    if in_back or at_root:
                        try:
                            index_candidates.append((path, self.read_text(path)))
                        except OSError:
                            pass

            # Yield all the normal docs first.
            yield from docs

            # A canonical description = a description/promo doc that is NOT itself a store listing.
            has_canonical = any(
                d.doc_type in _DESCRIPTION_TYPES and not is_store_listing_txt(d.doc_path.name)
                for d in docs
            )

            # CONDITIONAL store-listing fallback (provenance-aware dedup): only when no canonical
            # description exists. A store-only project still yields its content (no blind spot);
            # when description.docx/description.md is present, the derived store-txt is suppressed.
            store_listing_yielded = False
            if not has_canonical:
                for path, rel_parts, text in store_listing_candidates:
                    yield self._mk(project_id, path,
                                   _doc_type_for(rel_parts, path.name), "txt", text, project_dir)
                    store_listing_yielded = True

            # CONDITIONAL index.* fallback: last resort — only if NEITHER a canonical description
            # NOR a store listing provided product copy. Closes the coverage gap for the ~handful
            # of layouts whose index.php was the sole product source, without redundant copy.
            if not has_canonical and not store_listing_yielded:
                for path, raw in index_candidates:
                    visible = strip_markup(raw)
                    if visible:
                        yield self._mk(project_id, path, "promo", path.suffix.lstrip("."),
                                       visible, project_dir)
                        break  # one landing page is enough

    def declared_facets(self):
        """(#36) This corpus's declared structured facets (sources/sample_facts)."""
        return sample_declared_facets()

    def harvest_facts(self, project_id: str, project_dir: Path):
        """(#36) Structured facts from this project's back/config.yaml descriptor. The concrete
        field whitelist + secret handling live in sources/sample_facts (owned by this adapter)."""
        return harvest_facts_for(project_id, project_dir)

    def classification_policy(self) -> ClassificationPolicy:
        """(#37) This corpus's declared allow_ext (incl. `pdf`, #39) + file rules."""
        return atlas_classification_policy()

    def _mk(self, project_id, path, doc_type, ext, text, project_dir) -> SourceDoc:
        return SourceDoc(
            project_id=project_id,
            source_set=self.source_set,
            doc_path=path,
            doc_type=doc_type,
            ext=ext,
            raw_text=text,
            folder_meta={"project_dir": project_dir.name},
        )
