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
from rageval.extract.tabular import CsvReader, TabularLimits, read_tabular
from rageval.facts import PROVENANCE_TABULAR, FacetSpec, StructuredFact
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
                facts.append(StructuredFact(row.get("Claim"), facet, value,
                                            provenance=PROVENANCE_TABULAR))
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


# ===========================================================================
# CRITICAL-3 — single-column sheets are NOT silently dropped.
# ===========================================================================

def test_single_column_sheet_yields_rows_not_dropped(tmp_path):
    """CRITICAL-3: a genuinely 1-column sheet (claim ids) MUST yield its rows — the old ≥2-cell
    header rule dropped it silently (0 rows, no warning). Per the shape rule, with no WIDER row to
    mark a banner, the first non-blank row is the header and its single column is KEPT — the DATA is
    never lost (the essential property)."""
    csvf = tmp_path / "ids.csv"
    csvf.write_text("Claim\nC-1\nC-2\nC-3\n", encoding="utf-8")
    data = read_tabular(csvf)
    assert data.headers == ("Claim",)
    assert [r.get("Claim") for r in data.rows] == ["C-1", "C-2", "C-3"]


def test_single_column_xlsx_yields_rows_not_dropped(tmp_path):
    """xlsx path: a 1-column sheet's data rows are all present (not silently dropped)."""
    from openpyxl import Workbook

    xlsx = tmp_path / "ids.xlsx"
    wb = Workbook(); ws = wb.active; ws.title = "IDs"
    for v in ["Policy", "HH-1", "HH-2"]:
        ws.append([v])
    wb.save(str(xlsx))
    data = read_tabular(xlsx)
    assert data.headers == ("Policy",)
    assert [r.get("Policy") for r in data.rows] == ["HH-1", "HH-2"]


def test_single_column_with_banner_keeps_all_data(tmp_path):
    """A 1-column sheet WITH a leading banner: shape can't distinguish banner from header (no wider
    row), so per the spec the banner becomes the header — but NO data row is lost (the guarantee)."""
    csvf = tmp_path / "banner.csv"
    csvf.write_text("SYNTHETIC note\nC-1\nC-2\n", encoding="utf-8")
    data = read_tabular(csvf)
    # Every DATA value is retained (the anti-silent-drop property); header is the first cell.
    all_values = [next(iter(r.cells.values())) for r in data.rows]
    assert "C-1" in all_values and "C-2" in all_values


# ===========================================================================
# CRITICAL-4 — duplicate / blank entity_id never silently overwrites.
# ===========================================================================

def test_duplicate_entity_id_rows_do_not_collide(sheet_corpus, tmp_path):
    """Two rows with the SAME entity_id become TWO distinct sidecar records (disambiguated by row
    provenance) — a count/group_by can't silently under-report. (sheet_corpus registers the adapter
    so `peril` is a declared facet.)"""
    ents = [
        HarvestedEntity("C-1", "sheetcorp",
                        facts=(StructuredFact("C-1", "peril", "flood", PROVENANCE_TABULAR),),
                        provenance="loss-run.xlsx:S#1"),
        HarvestedEntity("C-1", "sheetcorp",   # SAME id, different row
                        facts=(StructuredFact("C-1", "peril", "fire", PROVENANCE_TABULAR),),
                        provenance="loss-run.xlsx:S#2"),
    ]
    records = entities_to_records(ents)
    assert len(records) == 2                       # both survive
    keys = {r.key for r in records}
    assert len(keys) == 2                          # distinct sidecar keys
    db = tmp_path / "dup.sqlite"
    conn = connect(db)
    for rec in records:
        upsert_project(conn, rec)
    got = list(all_projects(conn))
    conn.close()
    assert len(got) == 2
    perils = sorted(r.fact("peril") for r in got)
    assert perils == ["fire", "flood"]            # neither row clobbered the other


def test_blank_or_none_entity_id_is_skipped(tmp_path):
    """An empty/None entity_id is skipped with reason — never stored as project_id='None'."""
    ents = [
        HarvestedEntity("", "sheetcorp",
                        facts=(StructuredFact("", "peril", "hail", PROVENANCE_TABULAR),)),
        HarvestedEntity(None, "sheetcorp",       # type: ignore[arg-type]
                        facts=(StructuredFact("x", "peril", "wind", PROVENANCE_TABULAR),)),
        HarvestedEntity("C-9", "sheetcorp",
                        facts=(StructuredFact("C-9", "peril", "flood", PROVENANCE_TABULAR),)),
    ]
    records = entities_to_records(ents)
    assert [r.project_id for r in records] == ["C-9"]   # the two blank ids are dropped


# ===========================================================================
# MAJOR-3 — DoS caps: streaming truncation is surfaced, not an OOM.
# ===========================================================================

def test_row_cap_truncates_and_flags(tmp_path):
    csvf = tmp_path / "big.csv"
    lines = ["A,B"] + [f"{i},x" for i in range(50)]
    csvf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    data = read_tabular(csvf, limits=TabularLimits(max_rows=10))
    assert data.row_count == 10
    assert data.meta["truncated"] is True


def test_byte_cap_truncates_and_flags(tmp_path):
    csvf = tmp_path / "wide.csv"
    csvf.write_text("A,B\n" + "1,2\n" * 1000, encoding="utf-8")
    data = read_tabular(csvf, limits=TabularLimits(max_bytes=20))
    assert data.meta["truncated"] is True


# ===========================================================================
# minor — merged-cell / short rows (None cells) don't crash and pad correctly.
# ===========================================================================

def test_merged_cell_none_values_are_handled(tmp_path):
    """A merged/short row surfaces as None-padded cells (openpyxl yields None for merged gaps)."""
    xlsx = tmp_path / "merged.xlsx"
    _write_xlsx(xlsx, "M", "banner", ["A", "B", "C"],
                [["a1", None, "c1"], ["a2", "b2", None]])
    data = read_tabular(xlsx)
    assert data.row_count == 2
    assert data.rows[0].get("B") is None and data.rows[0].get("C") == "c1"
    assert data.rows[1].get("C") is None
