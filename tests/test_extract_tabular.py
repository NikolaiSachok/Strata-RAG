"""Tests for spreadsheet ingestion (#41) — self-contained (no vendored corpus).

Two layers, both from SYNTHETIC in-test fixtures (a small xlsx via openpyxl + a csv written to
disk):

  1. The generic READER (rageval.extract.tabular): parses xlsx + csv into the same neutral shape —
     rows as ordered {header: value} with sheet+row PROVENANCE — carrying NO column vocabulary.

  2. The ADAPTER→FACT→AGGREGATE path: a synthetic adapter maps the spreadsheet's COLUMNS to its
     DECLARED facets (one row = one entity) via harvest_entities(); the rows land in the sidecar
     facet store and an AGGREGATION question answers from the sidecar — proving rows are queryable
     as facts, that column names live in the ADAPTER (not the core), and that raw table rows are
     never embedded.
"""

from __future__ import annotations

import pytest

from rageval import aggregate
from rageval.enrich import entities_to_records
from rageval.extract.tabular import CsvReader, read_tabular
from rageval.facts import FacetSpec, StructuredFact
from rageval.sidecar import all_projects, connect, upsert_project
from rageval.sources import registry
from rageval.sources.base import HarvestedEntity, SourceAdapter

openpyxl = pytest.importorskip("openpyxl")


def _write_xlsx(path, sheet: str, banner: str, header: list[str], rows: list[list]) -> None:
    """A tiny xlsx with a leading single-cell BANNER row above the real header (mirrors a common
    real-world layout) so the reader's banner-skipping is exercised."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append([banner])          # banner row (one cell)
    ws.append(header)            # header row
    for r in rows:
        ws.append(r)
    wb.save(str(path))


# ===========================================================================
# 1. The generic reader — neutral shape + provenance, no column vocabulary.
# ===========================================================================

def test_xlsx_reader_parses_rows_with_provenance(tmp_path):
    xlsx = tmp_path / "loss-run.xlsx"
    _write_xlsx(xlsx, "Loss Run", "SYNTHETIC (fictional)",
                ["Claim", "Peril", "Status", "Reserve"],
                [["C-1", "flood", "open", 1000],
                 ["C-2", "fire", "closed", 0],
                 ["C-3", "flood", "open", 2500]])
    data = read_tabular(xlsx)

    assert data.backend == "openpyxl"
    assert data.headers == ("Claim", "Peril", "Status", "Reserve")
    assert data.row_count == 3
    assert data.sheets == ("Loss Run",)
    # Row provenance: sheet name + 1-based data-row index (header + banner excluded).
    first = data.rows[0]
    assert first.sheet == "Loss Run" and first.index == 1
    assert first.get("Claim") == "C-1" and first.get("Peril") == "flood"
    assert data.rows[2].index == 3 and data.rows[2].get("Claim") == "C-3"


def test_csv_reader_parses_rows_with_provenance(tmp_path):
    csvf = tmp_path / "commission.csv"
    csvf.write_text(
        "SYNTHETIC (fictional)\n"
        "Agent,Region,Policies\n"
        "AG-001,Ireland,1\n"
        "AG-002,Germany,2\n",
        encoding="utf-8",
    )
    data = read_tabular(csvf)
    assert data.backend == "csv"
    assert data.headers == ("Agent", "Region", "Policies")
    assert data.row_count == 2
    assert data.rows[0].sheet == "commission"      # sheet == file stem for csv
    assert data.rows[1].get("Region") == "Germany"


def test_blank_and_short_rows_are_handled(tmp_path):
    xlsx = tmp_path / "sparse.xlsx"
    _write_xlsx(xlsx, "S", "banner", ["A", "B", "C"],
                [["x", "y", "z"], [None, None, None], ["only-a"]])
    data = read_tabular(xlsx)
    # The fully-blank row is dropped; the short row pads missing cells with None.
    assert data.row_count == 2
    assert data.rows[1].get("A") == "only-a" and data.rows[1].get("C") is None


def test_duplicate_and_empty_headers_are_disambiguated(tmp_path):
    xlsx = tmp_path / "dupe.xlsx"
    _write_xlsx(xlsx, "S", "banner", ["Name", "Name", ""],
                [["a", "b", "c"]])
    data = read_tabular(xlsx)
    # Duplicate 'Name' → 'Name' + 'Name_1'; blank → 'column_3'. Every column addressable.
    assert set(data.headers) == {"Name", "Name_1", "column_3"}
    assert data.rows[0].get("Name") == "a" and data.rows[0].get("Name_1") == "b"


def test_reader_selection_and_unknown_ext(tmp_path):
    # An explicit reader is honoured; an unsupported extension fails LOUD (never silently empty).
    csvf = tmp_path / "x.csv"
    csvf.write_text("banner\nA,B\n1,2\n", encoding="utf-8")
    assert read_tabular(csvf, reader=CsvReader()).headers == ("A", "B")
    with pytest.raises(ValueError):
        read_tabular(tmp_path / "nope.parquet")


# ===========================================================================
# 2. Adapter maps columns → declared facets; rows queryable via aggregate.
# ===========================================================================

# The column→facet mapping lives in the ADAPTER (this test corpus), NOT the engine core.
_COLUMN_TO_FACET = {"Peril": "peril", "Status": "claim_status", "Reserve": "reserve"}


class _SpreadsheetAdapter(SourceAdapter):
    """A synthetic corpus whose data is a spreadsheet: each ROW is a claim entity. Declares the
    facets its columns map to, and harvests one HarvestedEntity per row."""

    source_set = "sheetcorp"

    def discover(self):
        return ()  # no narrative documents — this corpus is purely tabular

    def declared_facets(self):
        return (
            FacetSpec("peril", "text", "cause of loss"),
            FacetSpec("claim_status", "text", "open|closed"),
            FacetSpec("reserve", "int", "reserve amount"),
        )

    def harvest_entities(self):
        xlsx = self.root / "loss-run.xlsx"
        if not xlsx.is_file():
            return
        data = read_tabular(xlsx)
        for row in data.rows:
            facts = []
            for column, facet in _COLUMN_TO_FACET.items():
                value = row.get(column)
                if value is None:
                    continue
                facts.append(StructuredFact(row.get("Claim"), facet, value, provenance="descriptor"))
            yield HarvestedEntity(
                entity_id=str(row.get("Claim")),
                source_set=self.source_set,
                facts=tuple(facts),
                provenance=f"{xlsx.name}:{row.sheet}#{row.index}",
            )


@pytest.fixture
def sheet_corpus(tmp_path):
    """Register the spreadsheet adapter and lay out its xlsx. The autouse _reset_registries fixture
    restores the global adapter registry after the test."""
    registry.register_adapter("sheetcorp", _SpreadsheetAdapter)
    root = tmp_path / "corpus" / "sheetcorp"
    root.mkdir(parents=True)
    _write_xlsx(root / "loss-run.xlsx", "Loss Run", "SYNTHETIC (fictional)",
                ["Claim", "Peril", "Status", "Reserve"],
                [["C-1", "flood", "open", 1000],
                 ["C-2", "fire", "closed", 0],
                 ["C-3", "flood", "open", 2500],
                 ["C-4", "flood", "closed", 0]])
    yield root


def test_spreadsheet_rows_land_as_facts_and_aggregate(sheet_corpus, tmp_path):
    """The #41 acceptance: each row → a facts-only sidecar record; an AGGREGATION question over the
    spreadsheet answers from the sidecar via the existing `aggregate` path."""
    adapter = _SpreadsheetAdapter(sheet_corpus)
    entities = list(adapter.harvest_entities())
    assert len(entities) == 4  # one entity per data row

    records = entities_to_records(entities)
    db = tmp_path / "side.sqlite"
    conn = connect(db)
    for rec in records:
        upsert_project(conn, rec)
    conn.close()

    # The mapped columns are queryable FACETS now (declared by the adapter, not core columns).
    assert {"peril", "claim_status", "reserve"} <= aggregate.allowed_fields()

    # group_by over a spreadsheet column answers deterministically from the sidecar.
    res = aggregate.execute("group_by_count", field="peril", sidecar_path=db)
    counts = {r["peril"]: r["count"] for r in res.rows}
    assert counts == {"flood": 3, "fire": 1}

    # filter + count (open flood claims).
    res = aggregate.execute("count", filter={"peril": "flood", "claim_status": "open"},
                            sidecar_path=db)
    assert res.rows[0]["count"] == 2

    # lookup one row by its entity key; each row is its own entity.
    res = aggregate.execute("lookup", field="reserve", filter={"key": "sheetcorp/C-3"},
                            sidecar_path=db)
    assert res.rows[0]["reserve"] == "2500"

    # Provenance retained on the entity (file:sheet#row).
    assert entities[0].provenance.startswith("loss-run.xlsx:Loss Run#")


def test_row_entities_have_zero_chunks_never_embedded(sheet_corpus, tmp_path):
    """Rows are STRUCTURED facts, not prose: each record has chunk_count 0 (never embedded), so raw
    table rows can't dilute top-k retrieval."""
    adapter = _SpreadsheetAdapter(sheet_corpus)
    records = entities_to_records(list(adapter.harvest_entities()))
    db = tmp_path / "side.sqlite"
    conn = connect(db)
    for rec in records:
        upsert_project(conn, rec)
    got = {r.key: r for r in all_projects(conn)}
    conn.close()
    assert all(r.chunk_count == 0 for r in got.values())
    assert got["sheetcorp/C-1"].fact("peril") == "flood"
    assert got["sheetcorp/C-1"].fact("reserve") == 1000   # coerced to int per the declared type


def test_undeclared_column_is_not_stored(sheet_corpus, tmp_path):
    """Fail-closed: a fact for a column the adapter did NOT declare as a facet is never stored (the
    core knows no column names — only DECLARED facets are writable)."""
    ent = HarvestedEntity(entity_id="X-1", source_set="sheetcorp",
                          facts=(StructuredFact("X-1", "peril", "hail", "descriptor"),
                                 StructuredFact("X-1", "secret_note", "leak", "descriptor")))
    records = entities_to_records([ent])
    db = tmp_path / "s.sqlite"
    conn = connect(db)
    upsert_project(conn, records[0])
    got = {r.key: r for r in all_projects(conn)}
    conn.close()
    assert got["sheetcorp/X-1"].fact("peril") == "hail"
    assert "secret_note" not in got["sheetcorp/X-1"].facts
