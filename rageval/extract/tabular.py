"""Spreadsheet reading (#41) — pluggable xlsx/csv readers that yield STRUCTURED, provenance-
carrying rows. Corpus-neutral: it parses rows/cells and knows NO column names.

WHY spreadsheets are handled COMPLETELY differently from documents (the load-bearing decision):
  A policy PDF is narrative → you EMBED it and retrieve passages. A loss-run spreadsheet is
  STRUCTURED, AGGREGATABLE data → the useful questions are "how many open claims?", "total reserve
  by peril", "which agent wrote the most policies?". A vector top-k CANNOT count, sum, or group by;
  and embedding table rows as prose DILUTES retrieval (dozens of near-identical "row" chunks crowd
  out the real narrative). So spreadsheet rows do NOT become embedded chunks. They become
  STRUCTURED FACTS in the sidecar (facts.py / the EAV store), where the existing `aggregate` path
  answers count/list/group_by/lookup deterministically. This is the same "structured data → SQL
  facet, not vector search" split the config.yaml harvest already uses — now for tabular files.

THE CORE / ADAPTER BOUNDARY (why no column name lives here):
  This module is the generic MECHANISM: open the file, find the header row, yield each data row as
  an ORDERED {column_name: cell_value} mapping WITH its sheet + 1-based row number. It has no
  opinion about what "Premium" or "Cause of loss" means — mapping a spreadsheet's columns to typed
  facets is the corpus adapter's business (`declared_facets()` / `harvest_facts()`), exactly like
  the descriptor whitelist. So the engine core carries zero corpus-specific column vocabulary
  (grep-provable), and a new corpus with a totally different sheet supplies its own mapping with no
  core edit.

PLUGGABILITY: xlsx via openpyxl (lazy-imported, read-only mode), csv via the stdlib. Both produce
the SAME neutral `TabularData` shape, so the adapter's harvester is reader-agnostic and a third
format (ods, parquet) is a new `TabularReader` subclass with nothing downstream changing.
"""

from __future__ import annotations

import abc
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TabularRow:
    """One data row, normalised to a neutral shape carrying its own provenance.

    Fields:
      sheet   — the sheet/tab name ("" or the file stem for single-sheet csv).
      index   — 1-based row number WITHIN the data rows (header excluded) — stable row provenance.
      cells   — ORDERED {header: value} for this row. Header names are taken verbatim from the
                file's header row; values are strings/None (numeric coercion is the ADAPTER's job,
                against its declared facet TYPES — the reader stays type-agnostic).
    """

    sheet: str
    index: int
    cells: dict[str, object]

    def get(self, column: str, default: object = None) -> object:
        return self.cells.get(column, default)


@dataclass(frozen=True)
class TabularData:
    """A parsed spreadsheet: its headers + data rows, plus which backend read it (auditability).

    A single flat list of rows across all sheets (each row carries its `sheet`), so an adapter can
    iterate rows uniformly and still know the sheet provenance."""

    headers: tuple[str, ...]
    rows: tuple[TabularRow, ...]
    backend: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def sheets(self) -> tuple[str, ...]:
        seen: list[str] = []
        for r in self.rows:
            if r.sheet not in seen:
                seen.append(r.sheet)
        return tuple(seen)


# A leading "banner" row (e.g. a "SYNTHETIC — …" provenance line) sits ABOVE the real header in some
# corpora. We do NOT special-case any corpus text; instead the header-detection heuristic below
# skips leading rows that look like a single-cell banner (one non-empty cell) before the true
# multi-column header. Corpus-neutral: it keys off SHAPE (a 1-cell row above a multi-cell row), not
# any specific words.
def _looks_like_header(cells: list[str]) -> bool:
    """A plausible header row: at least two non-empty cells (a real table has multiple columns)."""
    non_empty = [c for c in cells if str(c).strip()]
    return len(non_empty) >= 2


def _dedupe_headers(raw: list[str]) -> list[str]:
    """Normalise a header row to non-empty, unique column names (stable, deterministic).

    Blank headers become `column_<n>`; a duplicate name gets a `_<n>` suffix. This guarantees the
    per-row {header: value} mapping has no colliding/empty keys, so the adapter can address every
    column unambiguously."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, name in enumerate(raw):
        base = str(name).strip() if name is not None else ""
        if not base:
            base = f"column_{i + 1}"
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        out.append(base)
    return out


def _rows_from_matrix(sheet: str, matrix: list[list[object]]) -> tuple[list[str], list[TabularRow]]:
    """Turn a raw row matrix (list of cell-lists) → (headers, TabularRows).

    Finds the header row (the first row that looks like a real multi-column header, skipping a
    leading single-cell banner), then yields each subsequent non-empty row as a TabularRow keyed by
    the deduped headers. Extra trailing cells are dropped; short rows pad with None."""
    header_idx = None
    for i, row in enumerate(matrix):
        if _looks_like_header([("" if c is None else str(c)) for c in row]):
            header_idx = i
            break
    if header_idx is None:
        return [], []
    headers = _dedupe_headers([("" if c is None else str(c)) for c in matrix[header_idx]])
    rows: list[TabularRow] = []
    data_index = 0
    for row in matrix[header_idx + 1:]:
        # Skip an entirely blank row (common trailing filler in spreadsheets).
        if not any(str(c).strip() for c in row if c is not None):
            continue
        data_index += 1
        cells: dict[str, object] = {}
        for col_i, header in enumerate(headers):
            value = row[col_i] if col_i < len(row) else None
            if isinstance(value, str):
                value = value.strip() or None
            cells[header] = value
        rows.append(TabularRow(sheet=sheet, index=data_index, cells=cells))
    return headers, rows


class TabularReader(abc.ABC):
    """The swappable spreadsheet-reading interface. A backend implements `read`."""

    name: str = "base"

    @abc.abstractmethod
    def read(self, path: Path) -> TabularData:
        raise NotImplementedError


class XlsxReader(TabularReader):
    """xlsx via openpyxl in READ-ONLY, values-only mode (fast, low-memory, no formula eval).

    read_only=True streams rows without loading the whole workbook; data_only=True returns the last
    cached cell VALUES rather than formulae (we want data, not `=SUM(...)`). Every sheet is read;
    each row keeps its sheet name as provenance."""

    name = "openpyxl"

    def read(self, path: Path) -> TabularData:
        from openpyxl import load_workbook  # lazy: only paid when an xlsx is actually read

        wb = load_workbook(filename=str(path), read_only=True, data_only=True)
        all_headers: list[str] = []
        all_rows: list[TabularRow] = []
        try:
            for ws in wb.worksheets:
                matrix = [list(r) for r in ws.iter_rows(values_only=True)]
                headers, rows = _rows_from_matrix(ws.title, matrix)
                if headers and not all_headers:
                    all_headers = headers
                all_rows.extend(rows)
        finally:
            wb.close()  # read-only workbooks hold a file handle until closed
        return TabularData(headers=tuple(all_headers), rows=tuple(all_rows), backend=self.name,
                           meta={"sheet_count": len({r.sheet for r in all_rows})})


class CsvReader(TabularReader):
    """csv via the stdlib. Single logical sheet (named after the file stem). Sniffs the delimiter
    with a tolerant fallback to comma; decodes UTF-8 leniently."""

    name = "csv"

    def read(self, path: Path) -> TabularData:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        # Sniff the dialect from a sample; fall back to a plain comma dialect if sniffing fails
        # (a one-column banner line can defeat the sniffer — comma is the safe default).
        try:
            dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        matrix = [list(row) for row in csv.reader(io.StringIO(raw), dialect)]
        sheet = Path(path).stem
        headers, rows = _rows_from_matrix(sheet, matrix)
        return TabularData(headers=tuple(headers), rows=tuple(rows), backend=self.name,
                           meta={"sheet_count": 1})


# Extension → reader. The generic dispatch point; a new tabular format registers here.
_READERS: dict[str, TabularReader] = {
    "xlsx": XlsxReader(),
    "csv": CsvReader(),
}


def read_tabular(path: Path, *, reader: TabularReader | None = None) -> TabularData:
    """Read a spreadsheet by extension (xlsx/csv), or with an explicit `reader`. Raises ValueError
    for an unsupported extension so a mis-wired adapter fails loud rather than silently empty."""
    path = Path(path)
    if reader is not None:
        return reader.read(path)
    ext = path.suffix.lower().lstrip(".")
    chosen = _READERS.get(ext)
    if chosen is None:
        raise ValueError(f"no tabular reader for .{ext} (supported: {sorted(_READERS)})")
    return chosen.read(path)
