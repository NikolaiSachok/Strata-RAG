"""Multi-format extraction — pluggable readers that turn a binary document into text/rows.

Phase 4 (#39 PDF, #41 spreadsheets) widened the engine beyond the md/txt/docx text family.
Two NEW binary formats dominate real enterprise corpora and need format-specific parsing:

  * PDF          — the dominant document format (policies, letters, reports). `extract.pdf`
                   pulls born-digital text with PAGE PROVENANCE, and DETECTS a scanned /
                   no-text-layer PDF so it is flagged (a coverage warning) instead of
                   emitting empty chunks (the OCR path is separate work — the vision seam).
  * spreadsheets — xlsx / csv carry STRUCTURED, aggregatable rows. `extract.tabular` parses
                   rows/cells with ROW+SHEET provenance; the rows land as STRUCTURED FACTS
                   in the sidecar (not embedded prose), so an aggregation question answers
                   deterministically from the facet store rather than diluting top-k.

THE DESIGN PRINCIPLE (why this is a package, not two ad-hoc helpers):
  Each format sits behind a small, SWAPPABLE interface (`PdfExtractor`, `TabularReader`) with a
  default backend chosen for portability (pypdf / openpyxl / stdlib csv), lazy-imported so a
  deployment that never touches PDFs pays nothing. The CORE stays generic: it parses BYTES into a
  neutral shape (pages of text; rows of cells). It carries NO corpus-specific column names — the
  column→facet mapping is the ADAPTER's business (`declared_facets()` / `harvest_facts()`), exactly
  like every other corpus-specific decision in the engine.
"""

from __future__ import annotations

from .pdf import (
    PdfExtraction,
    PdfExtractor,
    PdfPage,
    PypdfExtractor,
    extract_pdf,
    get_pdf_extractor,
)
from .tabular import (
    CsvReader,
    TabularData,
    TabularReader,
    TabularRow,
    XlsxReader,
    read_tabular,
)

__all__ = [
    # PDF (#39)
    "PdfExtraction",
    "PdfExtractor",
    "PdfPage",
    "PypdfExtractor",
    "extract_pdf",
    "get_pdf_extractor",
    # spreadsheets (#41)
    "CsvReader",
    "TabularData",
    "TabularReader",
    "TabularRow",
    "XlsxReader",
    "read_tabular",
]
