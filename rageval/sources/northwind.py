"""NorthwindAdapter — the "description-first, flat-id" corpus shape.

This adapter models a corpus family where each project is a numeric-id folder
(`0001/`, `0002/`, ...) and product content lives in predictable places:
  - `promo/description.md`        — the store/promo description
  - `docs/*.md`                   — overview/spec docs
  - `original/description_*.md`   — original/localised store descriptions

It also DISCOVERS the obvious noise (a template `CHANGELOG.md`, an agent-authored
`implementation_plan.md`) — but note: it does NOT drop them. Discovery's job is to find
and read every candidate; classify.py decides relevance. (Models a real description-first
corpus layout, genericised.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import ClassificationPolicy, SourceAdapter, SourceDoc
from .sample_facts import harvest_facts_for, sample_declared_facets
from .sample_policy import (
    docs_txt_doc_type,
    is_metadata_only_file,
    sample_classification_policy,
)

# Map filename/location patterns to a coarse doc_type. The classifier reasons partly off
# doc_type, so naming it here (cheaply, from the path) is useful even for noise.
_TEXT_EXTS = {".md", ".txt"}


def _doc_type_for(rel_parts: tuple[str, ...], name: str) -> str:
    low = name.lower()
    ext = low.rsplit(".", 1)[-1] if "." in low else ""
    # settings.md = structured METADATA → 'metadata' (enriched, not embedded). Checked first so
    # it never falls through to a 'spec'/'other' that would put it in the vector index.
    if is_metadata_only_file(low):
        return "metadata"
    if low in ("changelog.md", "changelog.txt"):
        return "changelog"
    if "implementation_plan" in low or low in ("task.md", "plan.md"):
        return "plan"
    if low in ("claude.md",):
        return "agent_doc"
    if "promo" in rel_parts and low.startswith("description"):
        return "promo"
    # promo/*.txt is real STORE COPY → keep it as promo content. (Path-aware: the audit
    # found promo/*.txt is product copy, while docs/*.txt is config/credential dumps.)
    if "promo" in rel_parts and ext == "txt":
        return "promo"
    if "original" in rel_parts and low.startswith("description"):
        return "description"
    # docs/*.txt: most are config/credential/Figma-link dumps (→ 'config', dropped), but a few
    # are real CONTENT (description.txt/ideas.txt/design.txt). Disambiguate by filename so we
    # stop dropping real product copy as a false negative. (One shared rule in base.py.)
    if "docs" in rel_parts and ext == "txt":
        return docs_txt_doc_type(low)
    if "docs" in rel_parts:
        return "spec"
    if low.startswith("description"):
        return "description"
    if low == "readme.md":
        return "readme"
    return "other"


class NorthwindAdapter(SourceAdapter):
    source_set = "northwind"

    def discover(self) -> Iterable[SourceDoc]:
        if not self.root.exists():
            return
        # Each immediate subdirectory of the root is one project.
        for project_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            project_id = project_dir.name
            for path in sorted(project_dir.rglob("*")):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                if ext not in _TEXT_EXTS:
                    continue  # .yaml/.png/etc. are not candidate *content* docs
                rel_parts = path.relative_to(project_dir).parts[:-1]  # dirs above the file
                try:
                    text = self.read_text(path)
                except OSError:
                    continue
                yield SourceDoc(
                    project_id=project_id,
                    source_set=self.source_set,
                    doc_path=path,
                    doc_type=_doc_type_for(rel_parts, path.name),
                    ext=ext.lstrip("."),
                    raw_text=text,
                    folder_meta={"project_dir": project_dir.name},
                )

    def declared_facets(self):
        """(#36) This corpus's declared structured facets (sources/sample_facts)."""
        return sample_declared_facets()

    def harvest_facts(self, project_id: str, project_dir: Path):
        """(#36) Structured facts from this project's back/config.yaml descriptor. The concrete
        field whitelist + secret handling live in sources/sample_facts (owned by this adapter)."""
        return harvest_facts_for(project_id, project_dir)

    def classification_policy(self) -> ClassificationPolicy:
        """(#37) This corpus's declared allow_ext + file rules (sources/sample_policy)."""
        return sample_classification_policy()
