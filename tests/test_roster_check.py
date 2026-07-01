"""Tests for the roster reconciliation tool.

Pure/deterministic over (roster rows, sidecar records) → fully unit-testable. We use a SYNTHETIC
TSV (written to a temp file) and synthetic ProjectRecords — all fictional. This locks down the
four statuses (MATCH / MISMATCH / sidecar-missing / tsv-missing) and the normalisation rules
(parenthetical notes, "must not appear" caveats, case/punctuation insensitivity).
"""

from __future__ import annotations

from rageval.roster_check import (
    RosterFinding,
    counts,
    load_tsv,
    normalize_publisher,
    reconcile,
    run,
)
from tests._helpers import make_record


# --- normalisation ----------------------------------------------------------

def test_normalize_case_space_punct_insensitive():
    assert normalize_publisher("Bowl-Master!!") == normalize_publisher("bowl master")
    assert normalize_publisher("  Pixel   Puzzler  ") == "pixel puzzler"


def test_normalize_drops_parenthetical_and_caveats():
    assert normalize_publisher("BowlMaster (internal note)") == "bowlmaster"
    assert normalize_publisher("BowlMaster - MUST NOT appear in the app") == "bowlmaster"
    assert normalize_publisher("GardenGrid (do not show); must not be visible") == "gardengrid"


def test_normalize_empty():
    assert normalize_publisher(None) == ""
    assert normalize_publisher("") == ""


# --- TSV loading ------------------------------------------------------------

def _write_tsv(tmp_path, name, rows):
    p = tmp_path / name
    lines = ["№\tID\tPublisher\tBundle"]  # № header
    for pid, bundle_id, publisher, bundle in rows:
        lines.append(f"{pid}\t{bundle_id}\t{publisher}\t{bundle}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_load_tsv_keys_on_number_column(tmp_path):
    p = _write_tsv(tmp_path, "fake.tsv", [
        ("2285", "bowltr01", "Maple Lagoon", "com.bowl.master"),
        ("2290", "gardtr02", "Copper Harbor", "com.garden.grid"),
    ])
    rows = load_tsv(p)
    assert [r.project_id for r in rows] == ["2285", "2290"]
    assert rows[0].publisher == "Maple Lagoon" and rows[0].source == "fake"


def test_load_tsv_tolerates_blank_and_short_rows(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text("№\tID\tPublisher\tBundle\n\n2285\tx\tMaple Lagoon\n", encoding="utf-8")
    rows = load_tsv(p)
    assert len(rows) == 1 and rows[0].publisher == "Maple Lagoon"


# --- reconciliation ---------------------------------------------------------

def _rec(source_set, pid, title):
    # roster_check reconciles the TSV's authoritative publisher against the sidecar's doc-inferred
    # app_name (the surface product title), so the sidecar side is set via app_name.
    return make_record(project_id=pid, source_set=source_set, app_name=title)


def _row(pid, publisher, source="northwind"):
    return type("R", (), {"project_id": pid, "publisher": publisher, "source": source})


def test_reconcile_all_four_statuses():
    tsv = [
        # in sidecar, same value → MATCH
        _row("2285", "Maple Lagoon"),
        # in sidecar, different value → MISMATCH (candidate)
        _row("2290", "Copper Harbor"),
        # sidecar has no title → sidecar-missing
        _row("1500", "Velvet Summit"),
        # not in sidecar at all → sidecar-missing
        _row("9999", "Amber Hollow"),
    ]
    sidecar = [
        _rec("northwind", "2285", "Maple Lagoon"),
        _rec("northwind", "2290", "OrchardGrid"),   # differs from TSV
        _rec("northwind", "1500", None),            # not stated in docs
        _rec("northwind", "0001", "InferredOnly"),  # not in TSV → tsv-missing
    ]
    findings = reconcile(tsv, sidecar)
    by_pid = {(f.project_id, f.sidecar_key): f.status for f in findings}
    assert by_pid[("2285", "northwind/2285")] == "MATCH"
    assert by_pid[("2290", "northwind/2290")] == "MISMATCH"
    assert by_pid[("1500", "northwind/1500")] == "sidecar-missing"
    assert by_pid[("9999", "")] == "sidecar-missing"
    assert by_pid[("0001", "northwind/0001")] == "tsv-missing"


def test_mismatch_ignores_caveats_and_case():
    tsv = [_row("2285", "Maple Lagoon (note); MUST NOT appear")]
    sidecar = [_rec("northwind", "2285", "maple-lagoon")]
    findings = reconcile(tsv, sidecar)
    assert findings[0].status == "MATCH"  # normalisation makes them equal


def test_project_id_under_multiple_source_sets():
    # Same numeric id can exist under northwind AND northwind-extra; both get checked.
    tsv = [_row("1801", "Velvet Summit")]
    sidecar = [_rec("northwind-extra", "1801", "Velvet Summit"),
               _rec("northwind", "1801", "DifferentTitle")]
    findings = reconcile(tsv, sidecar)
    statuses = {f.sidecar_key: f.status for f in findings}
    assert statuses["northwind-extra/1801"] == "MATCH"
    assert statuses["northwind/1801"] == "MISMATCH"


def test_run_end_to_end_with_real_tsv_file(tmp_path):
    p = _write_tsv(tmp_path, "northwind.tsv", [
        ("2285", "x", "Maple Lagoon", "com.b"),
        ("2290", "y", "StalePublisher", "com.g"),
    ])
    sidecar = [_rec("northwind", "2285", "Maple Lagoon"), _rec("northwind", "2290", "Copper Harbor")]
    findings = run([p], sidecar)
    c = counts(findings)
    assert c["MATCH"] == 1 and c["MISMATCH"] == 1


def test_findings_are_structured():
    f = RosterFinding("2285", "northwind", "Maple Lagoon", "northwind/2285", "Maple Lagoon", "MATCH")
    assert f.status == "MATCH" and f.project_id == "2285"
