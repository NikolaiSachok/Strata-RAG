"""End-to-end PDF ingestion wiring (#39): the atlas adapter reads a born-digital PDF into a
SourceDoc, the classifier includes it (per the atlas per-corpus `pdf` allow_ext), and a scanned
(no-text-layer) PDF is surfaced as a manifest COVERAGE WARNING — never a silent blind spot.

Self-contained: PDFs are generated in-test (reportlab for born-digital; pypdf blank pages for the
no-text-layer case). Uses a temp atlas-shaped corpus root so the committed sample corpus (which
ships no PDFs) is untouched.
"""

from __future__ import annotations

import pytest

from rageval.classify import CorpusRules, PolicyResolver, classify
from rageval.manifest import build_manifest
from rageval.sources.atlas import AtlasAdapter

pytest.importorskip("reportlab")


def _born_digital_pdf(path, text: str) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    y = 720
    for line in text.split("\n"):
        c.drawString(72, y, line)
        y -= 18
    c.showPage()
    c.save()


def _no_text_layer_pdf(path, n_pages: int = 2) -> None:
    from pypdf import PdfWriter

    w = PdfWriter()
    for _ in range(n_pages):
        w.add_blank_page(width=612, height=792)
    with open(path, "wb") as fh:
        w.write(fh)


@pytest.fixture
def atlas_pdf_corpus(tmp_path):
    """A temp atlas-shaped corpus: one project with a born-digital PDF, one with a scanned PDF."""
    root = tmp_path / "atlas"
    (root / "atlas-policy").mkdir(parents=True)
    (root / "atlas-scan").mkdir(parents=True)
    _born_digital_pdf(root / "atlas-policy" / "declarations.pdf",
                      "Household Policy Declarations\nPolicy number HH-1\nAnnual premium 960")
    _no_text_layer_pdf(root / "atlas-scan" / "settlement-letter.pdf", n_pages=2)
    return root


def test_atlas_reads_born_digital_pdf_with_page_provenance(atlas_pdf_corpus):
    docs = list(AtlasAdapter(atlas_pdf_corpus).discover())
    pdfs = [d for d in docs if d.ext == "pdf"]
    born = [d for d in pdfs if d.project_id == "atlas-policy"]
    assert born, "born-digital PDF should be discovered"
    doc = born[0]
    assert "Policy number HH-1" in doc.raw_text
    assert "[page 1]" in doc.raw_text                    # page provenance carried into the text
    assert not doc.folder_meta.get("pdf_scanned")


def test_atlas_scanned_pdf_is_flagged_not_dropped(atlas_pdf_corpus):
    docs = list(AtlasAdapter(atlas_pdf_corpus).discover())
    scan = [d for d in docs if d.project_id == "atlas-scan" and d.ext == "pdf"]
    assert scan, "scanned PDF must still be DISCOVERED (never silently skipped)"
    assert scan[0].folder_meta.get("pdf_scanned") is True
    assert scan[0].raw_text.strip() == ""                # no fabricated text


def test_born_digital_pdf_is_classified_included(atlas_pdf_corpus):
    """The atlas per-corpus allow_ext includes `pdf`, so a born-digital PDF is INCLUDED."""
    docs = list(AtlasAdapter(atlas_pdf_corpus).discover())
    rules = CorpusRules.load()
    policy = PolicyResolver().policy_for("atlas")
    born = next(d for d in docs if d.project_id == "atlas-policy" and d.ext == "pdf")
    dec = classify(born, rules, policy)
    assert dec.include and dec.reason == "ok"


def test_pdf_allow_ext_is_per_corpus(atlas_pdf_corpus):
    """`pdf` is declared by the ATLAS policy only — it must not become allowed for northwind."""
    from rageval.sources.base import SourceDoc
    from pathlib import Path

    other = SourceDoc(project_id="0001", source_set="northwind",
                      doc_path=Path("northwind") / "0001" / "x.pdf",
                      doc_type="description", ext="pdf", raw_text="x" * 200)
    dec = classify(other, CorpusRules.load(), PolicyResolver().policy_for("northwind"))
    assert not dec.include and "ext not allowed" in dec.reason


def test_manifest_surfaces_scanned_pdf_as_coverage_warning(atlas_pdf_corpus):
    docs = list(AtlasAdapter(atlas_pdf_corpus).discover())
    m = build_manifest(docs, CorpusRules.load())
    # The scanned PDF appears as a coverage warning (needs OCR), by doc_id.
    assert any("settlement-letter.pdf" in x for x in m.coverage.scanned_pdfs)
    # The born-digital PDF produced real chunks (it was embedded-eligible).
    assert m.total_chunks > 0
