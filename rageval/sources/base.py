"""The `SourceDoc` record and the abstract `SourceAdapter` interface.

These two types are the contract between "where documents come from" (corpus-specific,
behind an adapter) and "what we do with them" (corpus-agnostic: classify, chunk, embed,
index). Keep this module tiny and dependency-light — it's the shape everything else
agrees on.

PHASE-4 (corpus-agnostic engine): the adapter contract carries THREE optional hooks so ALL
corpus-specific knowledge lives behind the adapter, and a corpus that overrides none still
works on the generic defaults:

  * discover()             — walk the corpus → SourceDoc candidates (the only required method).
  * harvest_facts()        — (#36) yield StructuredFacts for the metadata sidecar. Default: none.
  * classification_policy()— (#37) declare per-corpus ingestion classification: the allowed
                             content extensions + filename/asset RULES (drop / metadata-only /
                             retype). Default: a generic policy that declares nothing extra.

The concrete filename heuristics for the bundled sample corpus (store-listing duplicates,
metadata-only files, docs/*.txt content-vs-config) used to live HERE as shared helpers; they are
now the sample adapters' declared POLICY (see sources/sample_policy.py). The core keeps only the
generic mechanism + defaults.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..facts import FacetSpec, StructuredFact


@dataclass(frozen=True)
class FileRule:
    """One adapter-declared classification rule matched against a discovered doc's FILENAME.

    The core provides the MECHANISM (a rule can drop a file, mark it metadata-only, or retype it);
    the adapter supplies the POLICY (which filenames, and what to do). A rule matches by exact
    filename (`name`, case-insensitive) OR by glob (`glob`, fnmatch). Exactly one action applies:

      action="drop"          — exclude the doc (reason carries the adapter's explanation).
      action="metadata_only" — INCLUDE but route to enrich, NOT the vector index (e.g. a
                               structured settings file). doc_type is optionally retyped too.
      action="retype"        — keep the doc but change its doc_type to `doc_type` (e.g. tag a
                               credential dump 'config' so a corpus rule drops it downstream).

    Kept corpus-neutral: the core never enumerates a filename; it just applies whatever rules an
    adapter declares. A corpus that declares none gets the generic default (nothing extra)."""
    action: str                      # "drop" | "metadata_only" | "retype"
    name: str | None = None          # exact filename match (case-insensitive)
    glob: str | None = None          # fnmatch glob (case-insensitive), e.g. "*_store.txt"
    doc_type: str | None = None      # for retype / metadata_only: the doc_type to assign
    reason: str = ""                 # human-readable reason surfaced in the manifest


@dataclass(frozen=True)
class ClassificationPolicy:
    """A corpus's ingestion-classification policy, declared by its adapter (#37).

    The engine ships a GENERIC default (no extra extensions, no file rules) so a corpus that
    declares nothing still classifies via corpus-rules.yaml alone. An adapter overrides
    `classification_policy()` to add:

      allow_ext  — content extensions this corpus contributes ON TOP OF the corpus-rules.yaml
                   baseline. Per-corpus, so adding a format for one corpus can never silently
                   change another (the classifier unions the adapter's set with the baseline for
                   THAT corpus's docs only).
      file_rules — adapter-declared FileRules (drop / metadata-only / retype by filename).

    Everything here is corpus-neutral in TYPE; the concrete values are the adapter's business."""
    allow_ext: frozenset[str] = frozenset()
    file_rules: tuple[FileRule, ...] = ()


@dataclass(frozen=True)
class HarvestedEntity:
    """A STRUCTURED, facts-only entity harvested from a system-of-record file (#41): a spreadsheet
    ROW, or any per-record descriptor whose value is its structured fields, NOT narrative prose.

    WHY this is a SEPARATE thing from a SourceDoc (the load-bearing distinction):
      A SourceDoc is NARRATIVE that gets chunked + embedded + retrieved. A HarvestedEntity is
      STRUCTURED data that gets stored in the sidecar facet store and answered by `aggregate`
      (count/group_by/sum) — it is NEVER embedded (embedding table rows dilutes top-k and can't be
      counted anyway). So the ingest path treats it as its OWN entity: one sidecar record per row,
      chunk_count 0, its columns mapped to DECLARED facets. This is what makes an aggregation
      question over a spreadsheet answer deterministically from the sidecar.

    Fields:
      entity_id   — a stable, per-corpus-UNIQUE id for this row/record (e.g. a claim id, or
                    "<file-stem>-r<row>"). Becomes the sidecar record's project_id, so it MUST be
                    unique within the source_set or two rows collapse onto one key.
      source_set  — which source-set/family this entity belongs to (the sidecar record's source_set).
      facts       — the StructuredFacts for this entity (each names a DECLARED facet; stored
                    fail-closed). The column→facet MAPPING is the adapter's business — the core
                    carries no column names.
      provenance  — where it came from, kept generic + human-readable (e.g. "loss-run.xlsx:Loss Run#3"
                    — file/sheet/row), retained for auditability.
    """

    entity_id: str
    source_set: str
    facts: tuple = ()
    provenance: str = ""


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

    # ---- optional Phase-4 hooks (generic defaults so a corpus can override none) --------

    def declared_facets(self) -> Iterable[FacetSpec]:
        """(#36/#40) Declare this corpus's structured FACETS — the positive, fail-closed allowlist
        of fields the adapter may emit as StructuredFacts, each with a value TYPE. This is the
        source of truth for BOTH storage (only a declared facet is stored) and queryability
        (aggregate validates a query field against the declared facets — NOT a core dataclass). The
        engine core enumerates NO facet name; a corpus declares its own. Default: no facets."""
        return ()

    def harvest_facts(self, project_id: str, project_dir: Path) -> Iterable[StructuredFact]:
        """(#36) Yield StructuredFacts for one project's metadata sidecar, from a per-project
        DESCRIPTOR (a config file / manifest / spreadsheet) that is NOT a narrative document and
        should never be embedded. The core consumes whatever facts an adapter emits and knows NO
        field names; the whitelist / fail-closed secret handling lives in the adapter's harvester
        (built on the reusable rageval.facts primitive). Only facts naming a DECLARED facet are
        stored (fail-closed). Default: no structured facts."""
        return ()

    def classification_policy(self) -> ClassificationPolicy:
        """(#37) Declare this corpus's ingestion-classification policy: the extra content
        extensions it contributes + its filename/asset FileRules (drop / metadata-only / retype).
        Default: a generic policy that declares nothing (the corpus classifies via
        corpus-rules.yaml alone)."""
        return ClassificationPolicy()

    def harvest_entities(self) -> Iterable["HarvestedEntity"]:
        """(#41) Yield STRUCTURED, facts-only entities — one per spreadsheet ROW (or any per-record
        system-of-record) — that must become their OWN sidecar records, NOT embedded prose.

        This is the spreadsheet-aggregation seam: the adapter uses the core `extract.read_tabular`
        reader to parse rows, maps each row's COLUMNS to its DECLARED facets (the mapping is the
        adapter's business — the core knows no column names), and yields one HarvestedEntity per
        row. The ingest path turns each into a facts-only ProjectRecord so `aggregate`
        (count/group_by/sum) answers over the rows from the sidecar. Corpus-wide (not per-project),
        because a spreadsheet spans the whole corpus, not one project folder. Default: none — a
        corpus with no tabular data overrides nothing and this path is a clean no-op."""
        return ()

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

    @staticmethod
    def read_pdf(path: Path):
        """Extract a born-digital PDF → a `PdfExtraction` (pages + text + a scanned/no-text-layer
        flag), via the pluggable extractor (#39).

        WHY here (next to read_docx): PDF is the dominant enterprise document format, so multi-format
        corpora need first-class PDF text. An adapter calls this, then builds a SourceDoc from
        `extraction.text()` (which carries `[page N]` provenance) — UNLESS `extraction.scanned` is
        True (no text layer → a scanned image PDF), which the adapter surfaces as a coverage warning
        rather than embedding empty chunks. Lazy import so a corpus that never sees a PDF pays
        nothing, and the pypdf dependency stays optional to the text-only path."""
        from ..extract.pdf import extract_pdf  # lazy: only paid when a PDF is actually read

        return extract_pdf(path)
